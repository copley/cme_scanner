from __future__ import annotations

import argparse
from typing import Dict

from ib_insync import Future
from loguru import logger

from config import load_config
from ib_client import IBClient
from indicators import compute_indicators
from model import TrendModel
from printer import print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend Continuation Probability App")
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file.')
    parser.add_argument('--host', default='127.0.0.1', help='TWS/IB Gateway host.')
    parser.add_argument('--port', default=7497, type=int, help='TWS/IB Gateway port.')
    parser.add_argument('--clientId', default=1, type=int, help='Client ID for IB connection.')
    args = parser.parse_args()

    config = load_config(args.config)

    log_cfg: Dict[str, object] = config.get('logging', {})
    log_level = log_cfg.get('level', 'INFO')
    log_file = log_cfg.get('file', 'trend_app.log')
    max_bytes = log_cfg.get('maxBytes', 10_000_000)
    backup_count = log_cfg.get('backupCount', 5)

    logger.remove()
    logger.add(log_file, rotation=max_bytes, retention=backup_count, level=log_level)
    logger.add(lambda msg: print(msg, end=""), level=log_level)

    ib_client = IBClient(host=args.host, port=args.port, clientId=args.clientId, logger=logger)

    model_path = config['model']['path']
    trend_model = TrendModel(model_path, logger=logger)

    symbols = config['symbols']
    contracts_cfg = config['contracts']
    timeframes = config['timeframes']
    thresholds = config['thresholds']

    results = []
    for symbol in symbols:
        contract_info = contracts_cfg.get(symbol)
        if not contract_info:
            logger.warning(f"No contract info found for symbol {symbol}, skipping.")
            continue

        contract = Future(
            symbol=contract_info['symbol'],
            lastTradeDateOrContractMonth=contract_info['lastTradeDateOrContractMonth'],
            exchange=contract_info['exchange'],
            currency=contract_info['currency'],
        )

        tf_indicator_dict: Dict[str, Dict[str, float]] = {}
        for tf in timeframes:
            tf_name = tf['name']
            durationStr = tf['durationStr']
            barSizeSetting = tf['barSizeSetting']
            lookback = tf.get('lookback', 100)
            try:
                df = ib_client.get_historical_data(
                    contract=contract,
                    durationStr=durationStr,
                    barSizeSetting=barSizeSetting,
                    whatToShow='TRADES',
                    useRTH=False,
                )
                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol} {tf_name}, skipping.")
                    continue
                indicators = compute_indicators(df, lookback=lookback)
                if indicators is None:
                    logger.warning(f"Not enough data to compute indicators for {symbol} {tf_name}.")
                    continue
                tf_indicator_dict[tf_name] = indicators
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to fetch or compute data for {symbol} {tf_name}: {exc}")

        if tf_indicator_dict:
            probability = trend_model.predict_probability(tf_indicator_dict)
            if probability >= thresholds['continue']:
                signal = 'Continue'
            elif probability >= thresholds['watch']:
                signal = 'Watch'
            else:
                signal = 'Reverse'
            for tf_name in tf_indicator_dict:
                row = {
                    'symbol': symbol,
                    'timeframe': tf_name,
                    'ADX': tf_indicator_dict[tf_name]['ADX'],
                    '%Above20MA': tf_indicator_dict[tf_name]['%Above20MA'],
                    'R2': tf_indicator_dict[tf_name]['R2'],
                    'probability': probability,
                    'signal': signal,
                }
                results.append(row)

    print_report(results, thresholds)
    ib_client.disconnect()


if __name__ == '__main__':
    main()
