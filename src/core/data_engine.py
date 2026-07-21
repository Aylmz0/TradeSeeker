import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import polars as pl
import requests
from loguru import logger

from src.core import constants


KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


class DataEngine:
    """Core Database Engine for TradeSeeker's ML Pipeline."""

    def __init__(self, db_path: str = "data/market_data.db"):
        """Initialize DataEngine with SQLite database path.

        Args:
            db_path: Path to SQLite database file (default: "data/market_data.db").
        """
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        """Create database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info("DataEngine: Created directory: {}", db_dir)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new SQLite connection with WAL mode and busy timeout.

        Returns:
            SQLite connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        return conn

    def _init_db(self):
        """Initialize database schema with market_data, decisions, and features tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp INTEGER,
                    coin TEXT,
                    interval TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (timestamp, coin, interval)
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    coin TEXT,
                    action TEXT,
                    ai_confidence REAL,
                    ml_probability REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_result REAL,
                    status TEXT
                )
            """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    timestamp INTEGER,
                    coin TEXT,
                    interval TEXT,
                    feature_json TEXT,
                    PRIMARY KEY (timestamp, coin, interval)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_coin_interval ON market_data (coin, interval)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions (status)")
            conn.commit()
            logger.info("DataEngine: Database schema initialized at '{}'.", self.db_path)
        except Exception as e:
            logger.error("DataEngine: Failed to initialize database: {}", e)
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_market_data(self, df: pl.DataFrame, coin: str, interval: str):
        """Log market data DataFrame to database.

        Args:
            df: DataFrame containing kline data.
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
        """
        self._insert_klines_bulk(df, coin, interval)

    def log_cycle_features(self, coin: str, interval: str, indicators: dict[str, Any]):
        """Log cycle features (indicators) to database.

        Args:
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
            indicators: Dictionary of indicator values to store.
        """
        if not isinstance(indicators, dict):
            return

        conn = self._get_connection()
        try:
            feature_data = {}
            for k, v in indicators.items():
                if isinstance(v, (int, float, str, bool)):
                    feature_data[k] = v
                elif isinstance(v, pl.Series):
                    feature_data[k] = v.to_list()[-1] if len(v) > 0 else None
                elif isinstance(v, list):
                    feature_data[k] = v[-1] if v else None

            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO features (timestamp, coin, interval, feature_json)
                   VALUES (?, ?, ?, ?)""",
                (int(time.time() * 1000), coin, interval, json.dumps(feature_data)),
            )
            conn.commit()
        except Exception as e:
            logger.error("DataEngine: Failed to log features for {}: {}", coin, e)
            conn.rollback()
        finally:
            conn.close()

    def _insert_klines_bulk(self, df: pl.DataFrame, coin: str, interval: str):
        """Bulk insert kline data into market_data table.

        Args:
            df: DataFrame containing kline data with standard columns.
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
        """
        if df.is_empty():
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            records = []

            ts_col = df["timestamp"]
            open_col = df["open"].to_list()
            high_col = df["high"].to_list()
            low_col = df["low"].to_list()
            close_col = df["close"].to_list()
            vol_col = df["volume"].to_list()

            for i in range(len(df)):
                ts_val = ts_col[i]
                if isinstance(ts_val, datetime):
                    ts_int = int(ts_val.replace(tzinfo=timezone.utc).timestamp() * 1000)
                else:
                    ts_int = int(ts_val)

                records.append(
                    (
                        ts_int,
                        coin,
                        interval,
                        float(open_col[i]),
                        float(high_col[i]),
                        float(low_col[i]),
                        float(close_col[i]),
                        float(vol_col[i]),
                    )
                )

            cursor.executemany(
                """
                INSERT OR IGNORE INTO market_data
                (timestamp, coin, interval, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            conn.commit()
            logger.info(
                "DataEngine: Ingested {} '{}' candles for {}.", len(records), interval, coin
            )
        except Exception as e:
            logger.error("DataEngine: Bulk insert failed for {} ({}): {}", coin, interval, e)
            conn.rollback()
            raise
        finally:
            conn.close()

    def _query_to_df(self, query: str, params: list | None = None) -> pl.DataFrame:
        """Execute SQL query and return results as Polars DataFrame.

        Args:
            query: SQL query string.
            params: Query parameters (optional).

        Returns:
            DataFrame with query results, or empty DataFrame if no rows.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params or [])
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame(schema=columns)
            return pl.DataFrame(rows, schema=columns, orient="row")
        finally:
            conn.close()

    def get_raw_market_data(
        self, coin: str, interval: str, limit: int | None = None
    ) -> pl.DataFrame:
        """Get raw market data from database.

        Args:
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
            limit: Maximum number of rows to return (optional).

        Returns:
            DataFrame with kline data ordered by timestamp ASC.
        """
        query = "SELECT * FROM market_data WHERE coin = ? AND interval = ? ORDER BY timestamp ASC"
        params: list = [coin, interval]
        if limit:
            query = """
                SELECT * FROM (
                    SELECT * FROM market_data
                    WHERE coin = ? AND interval = ?
                    ORDER BY timestamp DESC LIMIT ?
                ) ORDER BY timestamp ASC
            """
            params.append(limit)

        df = self._query_to_df(query, params)

        if not df.is_empty():
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("timestamp")
            )
        return df

    def wipe_market_data(self, coin: str | None = None, interval: str | None = None):
        """Wipe market data and features from database.

        Args:
            coin: Coin symbol to wipe (optional). If None, wipes all coins.
            interval: Interval to wipe (optional). If None, wipes all intervals.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if coin and interval:
                cursor.execute(
                    "DELETE FROM market_data WHERE coin = ? AND interval = ?", (coin, interval)
                )
                cursor.execute(
                    "DELETE FROM features WHERE coin = ? AND interval = ?", (coin, interval)
                )
                logger.info("DataEngine: Wiped data for {} ({}).", coin, interval)
            else:
                cursor.execute("DELETE FROM market_data")
                cursor.execute("DELETE FROM features")
                logger.info("DataEngine: Wiped ALL market data and features.")
            conn.commit()
        except Exception as e:
            logger.error("DataEngine: Wipe failed: {}", e)
            conn.rollback()
        finally:
            conn.close()

    def sync_bulk_history(self, coin: str, interval: str, target_count: int = 5000):
        """Sync historical kline data from Binance in batches of 1000.

        Args:
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
            target_count: Target number of candles to fetch (default: 5000).

        Returns:
            Total number of candles ingested.
        """
        import time as _time_module

        logger.info(
            f"[DataEngine] Starting Sync for {coin} ({interval}) -> Target: {target_count} candles"
        )

        base_url = "https://api.binance.com/api/v3/klines"
        symbol = f"{coin}USDT"
        total_ingested = 0
        end_time = None

        while total_ingested < target_count:
            try:
                params = {"symbol": symbol, "interval": interval, "limit": 1000}
                if end_time:
                    params["endTime"] = end_time - 1

                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                df = pl.DataFrame(data, schema=KLINE_COLUMNS)

                if df.is_empty():
                    break

                first_ts = int(df["timestamp"][0])
                end_time = first_ts

                self._insert_klines_bulk(df, coin, interval)
                total_ingested += len(df)

                logger.info(
                    f"[Sync] {coin} {interval}: Fetched batch of {len(df)}. Total: {total_ingested}/{target_count}"
                )

                if len(df) < 1000:
                    break

                _time_module.sleep(0.1)

            except Exception as e:
                logger.error("[Sync] Failed batch for {} {}: {}", coin, interval, e)
                break

        logger.info("[Sync] Completed for {} {}. Total candles: {}", coin, interval, total_ingested)
        return total_ingested

    def get_labeled_data(
        self,
        coin: str,
        interval: str,
        lookahead_periods: int = constants.ML_LOOKAHEAD_PERIODS,
        profit_threshold: float = constants.ML_DEFAULT_PROFIT_THRESHOLD
        if hasattr(constants, "ML_DEFAULT_PROFIT_THRESHOLD")
        else 0.003,
        loss_threshold: float = constants.ML_DEFAULT_LOSS_THRESHOLD
        if hasattr(constants, "ML_DEFAULT_LOSS_THRESHOLD")
        else -0.003,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Get labeled data for ML training.

        Creates target labels based on future returns vs dynamic thresholds
        derived from ATR.

        Args:
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
            lookahead_periods: Number of periods to look ahead for labeling.
            profit_threshold: Minimum return for BUY label (default: 0.003).
            loss_threshold: Maximum return for SELL label (default: -0.003).
            limit: Maximum number of rows to fetch (optional).

        Returns:
            DataFrame with labeled data including target_label column.
        """
        df = self.get_raw_market_data(coin, interval, limit=limit)
        if df.is_empty():
            return df

        df = df.with_columns(
            [
                pl.col("close").shift(-lookahead_periods).alias("future_close"),
            ]
        )
        df = df.with_columns(
            ((pl.col("future_close") - pl.col("close")) / pl.col("close")).alias("future_return")
        )

        tr0 = (pl.col("high") - pl.col("low")).abs()
        tr1 = (pl.col("high") - pl.col("close").shift()).abs()
        tr2 = (pl.col("low") - pl.col("close").shift()).abs()

        temp_df = df.select([tr0.alias("tr0"), tr1.alias("tr1"), tr2.alias("tr2")])
        tr = temp_df.select(pl.max_horizontal("tr0", "tr1", "tr2")).to_series()

        atr_14 = tr.ewm_mean(span=14, adjust=False)
        atr_pct = (atr_14 / df["close"]).fill_nan(constants.ML_ATR_LABEL_FLOOR)

        dynamic_profit = pl.max_horizontal(
            atr_pct * constants.ML_ATR_LABEL_MULTIPLIER,
            pl.Series([constants.ML_ATR_LABEL_FLOOR] * len(df)),
        )
        dynamic_loss = -dynamic_profit

        df = df.with_columns(
            [
                pl.when(pl.col("future_return") >= dynamic_profit)
                .then(1)
                .when(pl.col("future_return") <= dynamic_loss)
                .then(-1)
                .otherwise(0)
                .alias("target_label"),
            ]
        )

        df = df.filter(pl.col("future_return").is_not_null())

        logger.info(
            f"[Labeling] {coin} {interval} -> Created labels with {lookahead_periods} periods lookahead."
        )
        return df

    def fetch_and_store_klines(self, coin: str, interval: str, limit: int = 1000):
        """Fetch klines from Binance and store in database.

        Args:
            coin: Coin symbol (e.g. "XRP").
            interval: Candle interval (e.g. "3m").
            limit: Number of candles to fetch (default: 1000).

        Returns:
            True if successful, False otherwise.
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": f"{coin}USDT", "interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning("DataEngine: Received empty data for {} ({}).", coin, interval)
                return False

            df = pl.DataFrame(data, schema=KLINE_COLUMNS)

            for col in ["open", "high", "low", "close", "volume"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64).round(8))

            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime(time_unit="ms")).alias("timestamp")
            )

            df = df.fill_nan(None).drop_nulls()

            self._insert_klines_bulk(df, coin, interval)
            return True

        except Exception as e:
            logger.error(
                f"[DataEngine] Failed to fetch and store klines for {coin} ({interval}): {e}"
            )
            return False

    def log_decision_open(
        self,
        coin: str,
        direction: str,
        ai_confidence: float,
        ml_probability: float,
        entry_price: float,
    ) -> int | None:
        """Log an open trade decision to database.

        Args:
            coin: Coin symbol (e.g. "XRP").
            direction: Trade direction ("LONG" or "SHORT").
            ai_confidence: AI model confidence score.
            ml_probability: ML model probability score.
            entry_price: Entry price for the trade.

        Returns:
            Row ID of the inserted decision, or None on failure.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            cursor.execute(
                """INSERT INTO decisions
                   (timestamp, coin, action, ai_confidence, ml_probability, entry_price, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    now_ms,
                    coin,
                    direction.upper(),
                    round(float(ai_confidence), 4),
                    round(float(ml_probability), 4),
                    round(float(entry_price), 8),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(
                f"[DataEngine] Logged OPEN decision for {coin} ({direction}) -> row_id={row_id}"
            )
            return row_id
        except Exception as e:
            logger.error("DataEngine: Failed to log OPEN decision: {}", e)
            conn.rollback()
            return None
        finally:
            conn.close()

    def log_decision_close(
        self,
        coin: str,
        exit_price: float,
        pnl_result: float,
    ) -> bool:
        """Log a close trade decision to database.

        Updates the most recent OPEN decision for the coin with exit details.

        Args:
            coin: Coin symbol (e.g. "XRP").
            exit_price: Exit price for the trade.
            pnl_result: Profit/loss result in USDT.

        Returns:
            True if decision was closed, False if no OPEN decision found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE decisions
                   SET exit_price = ?, pnl_result = ?, status = 'CLOSED'
                   WHERE coin = ? AND status = 'OPEN'
                   ORDER BY timestamp DESC LIMIT 1""",
                (
                    round(float(exit_price), 8),
                    round(float(pnl_result), 4),
                    coin,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    f"[DataEngine] Closed decision for {coin}: exit=${exit_price:.4f}, PnL=${pnl_result:.2f}"
                )
                return True
            logger.warning("DataEngine: No OPEN decision found for {} to close.", coin)
            return False
        except Exception as e:
            logger.error("DataEngine: Failed to log CLOSE decision: {}", e)
            conn.rollback()
            return False
        finally:
            conn.close()


if __name__ == "__main__":
    import time

    logger.info("--- TradeSeeker Data Engine Ingestion Test ---")
    engine = DataEngine()

    test_coin = "XRP"
    intervals = ["3m", "15m", "1h"]

    for ival in intervals:
        logger.info("Backfilling 1000 candles for {} [{}]...", test_coin, ival)
        success = engine.fetch_and_store_klines(test_coin, interval=ival, limit=1000)
        if success:
            logger.success("Successfully written to SQLite (market_data table) for {}.", ival)
        else:
            logger.error("Failed to write {} data.", ival)
        time.sleep(1)

    logger.info("Testing Labeling Logic (Faz 1.3)...")
    df_labeled = engine.get_labeled_data("XRP", "15m")

    if not df_labeled.is_empty():
        logger.success("Labeling complete. Extracted {} labeled rows.", len(df_labeled))
        distribution = df_labeled.group_by("target_label").len().sort("target_label")
        logger.info("Label Distribution [1=BUY, -1=SELL, 0=HOLD]: {}", distribution)
        logger.info(
            "Demo Row: {}",
            df_labeled.select(
                ["timestamp", "close", "future_close", "future_return", "target_label"]
            ).tail(1),
        )
    else:
        logger.error("Labeling failed or empty dataframe.")

    logger.success("Phase 1.3 Labeling test completed.")
