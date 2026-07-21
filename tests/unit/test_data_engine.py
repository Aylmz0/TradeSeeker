import polars as pl
import pytest

from src.core.data_engine import DataEngine


def test_init_db(temp_db):
    engine = DataEngine(db_path=temp_db)
    conn = engine._get_connection()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    conn.close()
    assert "market_data" in tables
    assert "decisions" in tables
    assert "features" in tables


def test_log_market_data(temp_db, sample_ohlcv):
    engine = DataEngine(db_path=temp_db)
    engine.log_market_data(sample_ohlcv, coin="XRP", interval="3m")
    df = engine.get_raw_market_data("XRP", "3m")
    assert not df.is_empty()
    assert len(df) == 100


def test_get_raw_market_data(temp_db, sample_ohlcv):
    engine = DataEngine(db_path=temp_db)
    engine.log_market_data(sample_ohlcv, coin="XRP", interval="3m")
    df = engine.get_raw_market_data("XRP", "3m", limit=5)
    assert not df.is_empty()
    assert len(df) == 5
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}


def test_wipe_market_data(temp_db, sample_ohlcv):
    engine = DataEngine(db_path=temp_db)
    engine.log_market_data(sample_ohlcv, coin="XRP", interval="3m")
    assert not engine.get_raw_market_data("XRP", "3m").is_empty()
    engine.wipe_market_data("XRP", "3m")
    assert engine.get_raw_market_data("XRP", "3m").is_empty()


def test_log_decision_open(temp_db):
    engine = DataEngine(db_path=temp_db)
    row_id = engine.log_decision_open(
        coin="XRP",
        direction="LONG",
        ai_confidence=0.85,
        ml_probability=0.72,
        entry_price=2.50,
    )
    assert row_id is not None
    conn = engine._get_connection()
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    assert row["coin"] == "XRP"
    assert row["action"] == "LONG"
    assert row["status"] == "OPEN"
    assert row["entry_price"] == 2.50


def test_log_decision_close(temp_db):
    engine = DataEngine(db_path=temp_db)
    row_id = engine.log_decision_open(
        coin="XRP",
        direction="LONG",
        ai_confidence=0.85,
        ml_probability=0.72,
        entry_price=2.50,
    )
    result = engine.log_decision_close(coin="XRP", exit_price=2.60, pnl_result=10.0)
    assert result is True
    conn = engine._get_connection()
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    assert row["status"] == "CLOSED"
    assert row["exit_price"] == 2.60
    assert row["pnl_result"] == 10.0
