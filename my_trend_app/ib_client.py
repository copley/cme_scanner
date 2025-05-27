import time
from ib_insync import IB, util
from typing import Optional


class IBClient:
    """Wrapper around ib_insync.IB with retry logic."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,
        clientId: int = 1,
        max_retries: int = 3,
        logger: Optional[object] = None,
    ) -> None:
        self.ib = IB()
        self.host = host
        self.port = port
        self.clientId = clientId
        self.max_retries = max_retries
        self.logger = logger
        self.connect()

    def connect(self) -> None:
        attempt = 0
        while attempt < self.max_retries:
            try:
                if self.logger:
                    self.logger.info(
                        f"Connecting to IB at {self.host}:{self.port} (clientId={self.clientId})"
                    )
                self.ib.connect(self.host, self.port, clientId=self.clientId, timeout=10)
                if self.ib.isConnected():
                    if self.logger:
                        self.logger.info("Successfully connected to IB.")
                    return
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.error(f"Connection attempt {attempt+1} failed: {exc}")
                time.sleep(5)
                attempt += 1
        raise ConnectionError("Could not connect to TWS/IB Gateway after multiple retries.")

    def get_historical_data(
        self,
        contract,
        endDateTime: str = '',
        durationStr: str = '1 D',
        barSizeSetting: str = '1 day',
        whatToShow: str = 'TRADES',
        useRTH: bool = True,
    ):
        if not self.ib.isConnected():
            if self.logger:
                self.logger.warning("IB not connected, reconnecting...")
            self.connect()
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=1,
        )
        return util.df(bars)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
