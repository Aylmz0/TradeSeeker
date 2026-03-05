import logging
import os
import sqlite3
import pandas as pd
import requests
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataEngine:
    """
    Core Database Engine for TradeSeeker's ML Pipeline.
    Handles SQLite ingestion, storage, and retrieval of market data, ML features, and AI decisions.
    """

    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        """Ensure the directory for the database exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"[DataEngine] Created directory: {db_dir}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a configured SQLite connection."""
        # Check_same_thread=False allows multi-threaded access if needed, but we should handle locks.
        # Isolation level None means auto-commit mode, but we will explicitly commit.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn

    def _init_db(self):
        """Initialize the SQLite database schema for Phase 1.1."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 1. Market Data Table: Stores raw OHLCV from Binance (3m, 15m, 1h)
            cursor.execute('''
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
            ''')

            # 2. Decisions Table: The Feedback Loop (Self-Learning memory)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    coin TEXT,
                    action TEXT,          -- LONG, SHORT, WAIT
                    ai_confidence REAL,   -- The confidence score given by DeepSeek
                    ml_probability REAL,  -- The raw probability output from XGBoost
                    entry_price REAL,
                    exit_price REAL,
                    pnl_result REAL,      -- The actual realized PnL
                    status TEXT           -- OPEN, CLOSED
                )
            ''')

            # 3. Features Table: Normalized indicator values ready for XGBoost DMatrix
            # We use a JSON text field for feature_data to allow adding/removing indicators 
            # without requiring complex schema migrations (Schema Evolution flexibility).
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    timestamp INTEGER,
                    coin TEXT,
                    interval TEXT,
                    feature_json TEXT,
                    PRIMARY KEY (timestamp, coin, interval)
                )
            ''')
            
            # Create indexes for query performance (O(log N) lookups)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_coin_interval ON market_data (coin, interval)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions (status)')
            
            conn.commit()
            logger.info(f"[DataEngine] Database schema initialized successfully at '{self.db_path}'.")
            
        except Exception as e:
            logger.error(f"[DataEngine] Failed to initialize database: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def _insert_klines_bulk(self, df: pd.DataFrame, coin: str, interval: str):
        """Bulk insert a pandas DataFrame of KLines into SQLite."""
        if df.empty:
            return
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Convert timestamp to integer milliseconds for storage
            # The DF arriving from RealMarketData has 'timestamp' as datetime64
            records = []
            for _, row in df.iterrows():
                # Extract int timestamp
                ts_int = int(row['timestamp'].timestamp() * 1000)
                
                records.append((
                    ts_int,
                    coin,
                    interval,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ))
                
            # INSERT OR IGNORE protects against duplicate timestamp overlapping
            cursor.executemany('''
                INSERT OR IGNORE INTO market_data 
                (timestamp, coin, interval, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            logger.info(f"[DataEngine] Ingested {len(records)} '{interval}' candles for {coin}.")
        except Exception as e:
            logger.error(f"[DataEngine] Bulk insert failed for {coin} ({interval}): {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_raw_market_data(self, coin: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch raw market data from SQLite into a pandas DataFrame."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM market_data WHERE coin = ? AND interval = ? ORDER BY timestamp ASC"
            params = [coin, interval]
            if limit:
                # We need to get the "latest" N candles, so we sort DESC then reverse,
                # or just use a subquery. Subquery is cleaner.
                query = f"""
                    SELECT * FROM (
                        SELECT * FROM market_data 
                        WHERE coin = ? AND interval = ? 
                        ORDER BY timestamp DESC LIMIT ?
                    ) ORDER BY timestamp ASC
                """
                params.append(limit)
                
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            return df
        finally:
            conn.close()

    def get_labeled_data(
        self, 
        coin: str, 
        interval: str, 
        lookahead_periods: int = 5,
        profit_threshold: float = 0.005,  # 0.5% profit
        loss_threshold: float = -0.005,   # -0.5% loss
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Creates the ML targets (Labels) by looking ahead in time.
        Calculates the future percentage return from the current close to the close at (t + lookahead).
        Assigns:
        -  1 (BUY):  Return >= profit_threshold
        - -1 (SELL): Return <= loss_threshold
        -  0 (HOLD): Everything else (Choppy/Sideways)
        """
        df = self.get_raw_market_data(coin, interval, limit=limit)
        if df.empty:
            return df
            
        # Calculate future return using Vectorized shift (O(1) equivalent in pandas)
        # Shift with negative number moves future values backwards to the current row
        df['future_close'] = df['close'].shift(-lookahead_periods)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Apply labels using numpy.select for blazing fast vectorization
        conditions = [
            (df['future_return'] >= profit_threshold),
            (df['future_return'] <= loss_threshold)
        ]
        choices = [1, -1] # 1=BUY, -1=SELL
        
        # Default is 0 (HOLD)
        df['target_label'] = np.select(conditions, choices, default=0)
        
        # Drop rows with NaN future_return (the last 'lookahead_periods' rows have no future yet)
        df.dropna(subset=['future_return'], inplace=True)
        
        logger.info(f"[Labeling] {coin} {interval} -> Created labels with {lookahead_periods} periods lookahead.")
        return df

    def fetch_and_store_klines(self, coin: str, interval: str, limit: int = 1000):
        """
        Fetches KLines directly from Binance API to avoid importing bot configs during tests.
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": f"{coin}USDT", "interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"[DataEngine] Received empty data for {coin} ({interval}).")
                return False
                
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ],
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float).round(8)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Sanitize corrupted candles
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            self._insert_klines_bulk(df, coin, interval)
            return True
            
        except Exception as e:
            logger.error(f"[DataEngine] Failed to fetch and store klines for {coin} ({interval}): {e}")
            return False

    def log_decision_open(
        self,
        coin: str,
        direction: str,
        ai_confidence: float,
        ml_probability: float,
        entry_price: float
    ) -> Optional[int]:
        """Log an OPEN trade decision to the decisions table. Returns the row ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO decisions 
                   (timestamp, coin, action, ai_confidence, ml_probability, entry_price, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    int(pd.Timestamp.now().timestamp() * 1000),
                    coin,
                    direction.upper(),
                    round(float(ai_confidence), 4),
                    round(float(ml_probability), 4),
                    round(float(entry_price), 8),
                )
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"[DataEngine] Logged OPEN decision for {coin} ({direction}) -> row_id={row_id}")
            return row_id
        except Exception as e:
            logger.error(f"[DataEngine] Failed to log OPEN decision: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def log_decision_close(
        self,
        coin: str,
        exit_price: float,
        pnl_result: float
    ) -> bool:
        """Update the latest OPEN decision for a coin with exit price and realized PnL."""
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
                )
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"[DataEngine] Closed decision for {coin}: exit=${exit_price:.4f}, PnL=${pnl_result:.2f}")
                return True
            else:
                logger.warning(f"[DataEngine] No OPEN decision found for {coin} to close.")
                return False
        except Exception as e:
            logger.error(f"[DataEngine] Failed to log CLOSE decision: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()


if __name__ == "__main__":
    import time
    # Test execution for Faz 1.2
    print("--- TradeSeeker Data Engine Ingestion Test ---")
    engine = DataEngine()
    
    test_coin = "XRP"
    intervals = ["3m", "15m", "1h"]
    
    for ival in intervals:
        print(f"\n[Test] Backfilling 1000 candles for {test_coin} [{ival}]...")
        success = engine.fetch_and_store_klines(test_coin, interval=ival, limit=1000)
        if success:
            print(f"[OK] Successfully written to SQLite (market_data table) for {ival}.")
        else:
            print(f"[FAIL] Failed to write {ival} data.")
        time.sleep(1) # Sleep to avoid rate limits
        
    print("\n[Test] Testing Labeling Logic (Faz 1.3)...")
    df_labeled = engine.get_labeled_data("XRP", "15m", lookahead_periods=5, profit_threshold=0.005, loss_threshold=-0.005)
    
    if not df_labeled.empty:
        print(f"[OK] Labeling complete. Extracted {len(df_labeled)} labeled rows.")
        # Show distribution of labels
        distribution = df_labeled['target_label'].value_counts().to_dict()
        print(f"[INFO] Label Distribution [1=BUY, -1=SELL, 0=HOLD]: {distribution}")
        print("\nDemo Row:\n", df_labeled[['timestamp', 'close', 'future_close', 'future_return', 'target_label']].tail(1))
    else:
        print("[FAIL] Labeling failed or empty dataframe.")
        
    print("\n[OK] Phase 1.3 Labeling test completed.")

