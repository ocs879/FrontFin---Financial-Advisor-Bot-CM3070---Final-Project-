# frontfin_config.py

FINAL_CONFIG = {
    "version": "2025-08-30",
    "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # Tickers to monitor
    "prices": {
        "start": "2010-01-01",   # Start date for historical data
        "end": None,             # None = up to today
        "interval": "1d"         # Daily bars
    },
    "strategy": {
        "fast": 10,              # Fast SMA window
        "slow": 100,             # Slow SMA window
        "target_vol_annual": 0.15,  # Target annual volatility (15%)
        "trade_band": 0.02,         # Trade trigger band (2%)
        "hysteresis": 0.002,        # Whipsaw protection band (0.2%)
        "allow_short": False,       # Long-only mode
        "max_pos": 1.0              # Max 100% of equity in one asset
    },
    "exec": {
        "commission": 0.001,       # 0.1% commission
        "slippage": 0.0005,        # 0.05% slippage
        "initial_capital": 100_000 # Starting capital
    },
    "portfolio": {
        "top_k": 5,                # Pick top 5 signals
        "max_weight": 0.25         # 25% max per asset
    }
}
