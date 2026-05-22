-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Market Data (Ticks/Candles)
CREATE TABLE IF NOT EXISTS market_candles (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL, -- '1m', '5m', '1h'
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    source TEXT, -- 'binance', 'metatrader'
    UNIQUE (time, symbol, interval)
);

-- Convert to Hypertable (Partition by time)
SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);

-- Brain States (The "Neuroscope" History)
CREATE TABLE IF NOT EXISTS brain_states (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    run_id TEXT, -- Link to MLflow Run
    
    -- Predictions
    prob_up DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    
    -- Context
    trend_m5 INTEGER,
    trend_h1 INTEGER,
    
    -- Features (Stored as JSONB for flexibility or Array)
    features JSONB,
    
    -- Logic Log
    logic_log TEXT[],
    
    UNIQUE (time, symbol, run_id)
);

SELECT create_hypertable('brain_states', 'time', if_not_exists => TRUE);

-- Trades (Simulated or Real)
CREATE TABLE IF NOT EXISTS trades (
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    symbol TEXT NOT NULL,
    direction INTEGER, -- 1=LONG, -1=SHORT
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    status TEXT, -- 'OPEN', 'CLOSED'
    run_id TEXT
);

SELECT create_hypertable('trades', 'entry_time', if_not_exists => TRUE);
