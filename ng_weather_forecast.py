import os
from datetime import datetime, timedelta
from typing import List, Optional

import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_REQUESTS = False

try:
    import xarray as xr
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    xr = None
    np = None

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
except ImportError:  # pragma: no cover - optional dependency
    EClient = EWrapper = Contract = None


GFS_BASE_URL = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
    "gfs.{date}/{hour}/atmos/"
    "gfs.t{hour}z.pgrb2.0p25.f{fh:03d}"
)

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def download_gfs_file(model_date: str, model_hour: str, forecast_hour: int, dest: str) -> str:
    """Download a single GFS GRIB2 file."""
    if not HAS_REQUESTS:
        raise RuntimeError("requests is required for downloading")

    os.makedirs(dest, exist_ok=True)
    filename = f"gfs.t{model_hour}z.pgrb2.0p25.f{forecast_hour:03d}"
    url = GFS_BASE_URL.format(date=model_date, hour=model_hour, fh=forecast_hour)
    out_path = os.path.join(dest, filename)

    if os.path.exists(out_path):
        logging.info("Using cached %s", filename)
        return out_path

    logging.info("Downloading %s", url)
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return out_path


def fetch_forecast_series(model_date: str, model_hour: str, dest: str) -> List[str]:
    """Fetch GRIB2 files for forecast hours 3..120."""
    hours = list(range(3, 123, 3))  # 5-day window
    paths = []
    for fh in hours:
        try:
            path = download_gfs_file(model_date, model_hour, fh, dest)
            paths.append(path)
        except Exception as exc:  # pragma: no cover - network dependent
            logging.warning("Failed to download f%03d: %s", fh, exc)
    return paths


def extract_temperature(paths: List[str]):
    """Extract 2-meter temperature from GRIB files using xarray/cfgrib."""
    if xr is None:
        raise RuntimeError("xarray is required for parsing GRIB files")

    temps = []
    for p in paths:
        ds = xr.open_dataset(
            p,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "surface"},
        )
        if "t2m" in ds:
            var = ds["t2m"]
        elif "2t" in ds:
            var = ds["2t"]
        else:
            raise RuntimeError("No 2m temperature field found")
        temps.append(var)
    combined = xr.concat(temps, dim="time")
    return combined


def load_population_weights(path: str) -> "xr.DataArray":
    """Load population weights grid (NumPy .npy)."""
    if xr is None or np is None:
        raise RuntimeError("NumPy and xarray are required")
    arr = np.load(path)
    return xr.DataArray(arr)


def compute_degree_days(temps_k: "xr.DataArray", weights: "xr.DataArray"):
    """Compute HDD/CDD/GWDD using population weights."""
    if np is None:
        raise RuntimeError("NumPy is required")

    temps_f = temps_k - 273.15
    temps_f = temps_f * 9 / 5 + 32

    weighted = (temps_f * weights).sum(dim=("latitude", "longitude")) / weights.sum()
    daily_mean = weighted.resample(time="1D").mean()

    hdd = xr.where(65 - daily_mean > 0, 65 - daily_mean, 0)
    cdd = xr.where(daily_mean - 65 > 0, daily_mean - 65, 0)
    gwdd = xr.where(daily_mean >= 65, cdd, hdd)

    return daily_mean, hdd, cdd, gwdd


if EClient is not None:
    class IBClient(EWrapper, EClient):  # pragma: no cover - network dependent
        def __init__(self):
            EClient.__init__(self, self)
            self.price = None

        def tickPrice(self, reqId, tickType, price, attrib):
            if tickType == 4:  # Last price
                self.price = price


def fetch_ng_price(host: str = "127.0.0.1", port: int = 7496, client_id: int = 0) -> Optional[float]:
    """Retrieve current NG front-month price using IBKR."""
    if EClient is None:
        logging.warning("ibapi not installed; skipping price fetch")
        return None

    app = IBClient()
    app.connect(host, port, client_id)

    contract = Contract()
    contract.symbol = "NG"
    contract.secType = "FUT"
    contract.exchange = "NYMEX"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = ""  # front month

    app.reqMktData(1, contract, "", False, False, [])

    end = datetime.utcnow() + timedelta(seconds=10)
    while datetime.utcnow() < end and app.price is None:
        app.run()

    app.disconnect()
    return app.price


def estimate_price_impact(delta_gwdd: float) -> float:
    """Simple rule-based impact estimate in $/mmBtu."""
    return round(delta_gwdd * 0.012, 3)


def main(model_date: Optional[str] = None, model_hour: str = "00") -> None:
    model_date = model_date or datetime.utcnow().strftime("%Y%m%d")
    logging.info("Processing GFS run %s %sz", model_date, model_hour)

    data_paths = fetch_forecast_series(model_date, model_hour, dest="gfs_data")
    if not data_paths:
        logging.error("No forecast data downloaded; aborting")
        return

    temps = extract_temperature(data_paths)
    weights = load_population_weights("weights.npy")
    mean_t, hdd, cdd, gwdd = compute_degree_days(temps, weights)

    total_gwdd = float(gwdd.sum())
    logging.info("5-Day GWDD total: %.2f", total_gwdd)

    ng_price = fetch_ng_price()
    if ng_price is not None:
        impact = estimate_price_impact(total_gwdd)
        forecast_price = ng_price + impact
        print("\nNatural Gas Weather Forecast")
        print(f"Model: {model_date} {model_hour}z")
        print(f"GWDD (5d): {total_gwdd:.1f}")
        print(f"Current NG1!: ${ng_price:.3f}")
        print(f"Estimated Impact: ${impact:+.3f}")
        print(f"Forecast NG1!: ${forecast_price:.3f}")
    else:
        print("\nNatural Gas Weather Forecast")
        print(f"Model: {model_date} {model_hour}z")
        print(f"GWDD (5d): {total_gwdd:.1f}")
        print("NG price unavailable; IBAPI not connected")


if __name__ == "__main__":
    main()
