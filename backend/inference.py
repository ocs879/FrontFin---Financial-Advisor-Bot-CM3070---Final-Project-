# inference.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from frontfin_config import FINAL_CONFIG


def download_prices(symbols, start, end=None, interval="1d"):
    """
    Downloads historical OHLCV data for the given symbols.
    Handles both single-index and MultiIndex column formats.
    Returns a dict: symbol -> DataFrame with at least a 'Close' column.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    data = {}
    for s in symbols:
        try:
            df = yf.download(
                s,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                group_by="column",
                auto_adjust=False,
            )
            if df is None or df.empty:
                print(f"[WARN] {s}: empty download")
                continue

            # Ensure there's a 'Close' column
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]

            # Handle MultiIndex columns (rare with group_by="column")
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = ["_".join(map(str, col)).strip() for col in df.columns]
                df.columns = flat_cols

            df = df.dropna(how="all")
            df.index = pd.to_datetime(df.index)

            if "Close" not in df.columns:
                print(f"[WARN] {s}: missing 'Close' after adjustment; columns: {list(df.columns)}")
                continue

            data[s] = df
        except Exception as e:
            print(f"[WARN] {s}: download failed -> {e}")
    return data


def extract_close_series(df, sym):
    """
    Extracts a clean pd.Series of closing prices from varied DataFrame shapes:
    - MultiIndex column (like ('Close', 'AAPL'))
    - Flattened columns like 'Close_AAPL' or 'Adj Close'
    - Single-index with 'Close'
    - Single-column fallback
    """
    if isinstance(df.columns, pd.MultiIndex):
        for key in [("Close", sym), ("Adj Close", sym)]:
            if key in df.columns:
                s = df[key].dropna()
                s.name = "Close"
                return s

    candidates = [f"Close_{sym}", f"Adj Close_{sym}", "Close", "Adj Close"]
    for col in candidates:
        if col in df.columns:
            s = df[col].dropna()
            s.name = "Close"
            return s

    # If only one column remains, assume it's the close
    if df.shape[1] == 1:
        s = df.iloc[:, 0].dropna()
        s.name = "Close"
        return s

    raise ValueError(f"{sym}: unable to extract 'Close' series; columns: {list(df.columns)}")


def compute_position_series(
    close: pd.Series,
    fast: int,
    slow: int,
    target_vol_annual: float,
    band: float,
    hyst: float,
    allow_short: bool,
    max_pos: float,
):
    """
    Given a close price series, computes the target position series
    using SMA crossover, volatility scaling, hysteresis & trade band.
    Returns a pd.Series with positions (floats).
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)

    close = close.dropna()
    if close.ndim != 1 or len(close) < max(slow + 30, 80):
        raise ValueError(f"'close' must be 1-D series with enough rows (len={len(close)})")

    df = pd.DataFrame({"Close": close})
    df["SMA_Fast"] = df["Close"].rolling(fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(slow).mean()

    up = df["SMA_Fast"] > df["SMA_Slow"] * (1 + hyst)
    down = df["SMA_Fast"] < df["SMA_Slow"] * (1 - hyst)

    sig = pd.Series(0.0, index=df.index)
    sig[up] = 1.0
    if allow_short:
        sig[down] = -1.0

    daily_target = target_vol_annual / np.sqrt(252)
    vol = df["Close"].pct_change().rolling(20).std()
    size = (daily_target / (vol + 1e-12)).clip(upper=max_pos).fillna(0.0)

    pos = (sig * size).clip(-max_pos, max_pos)
    if not allow_short:
        pos = pos.clip(lower=0.0, upper=max_pos)

    return pos.ffill().fillna(0.0)


def get_recommendations(cash: float = None):
    """
    Main entry: returns top-K portfolio recommendations:
     - status: "ok" or "error"
     - recommendations: list of {'ticker','signal','price'}
     - portfolio: {'cash','weights'}
     - debug: optional error list (per symbol)
    """
    try:
        cfg = FINAL_CONFIG
        uni = cfg["universe"]
        scfg = cfg["strategy"]
        pcfg = cfg["portfolio"]
        cash = cash if cash is not None else cfg["exec"]["initial_capital"]

        start = (datetime.utcnow() - timedelta(days=500)).strftime("%Y-%m-%d")
        data = download_prices(uni, start, None, cfg["prices"]["interval"])

        recs = []
        errs = []
        for sym, df in data.items():
            try:
                close_ser = extract_close_series(df, sym)
                pos = compute_position_series(
                    close_ser,
                    fast=scfg["fast"],
                    slow=scfg["slow"],
                    target_vol_annual=scfg["target_vol_annual"],
                    band=scfg["trade_band"],
                    hyst=scfg["hysteresis"],
                    allow_short=scfg["allow_short"],
                    max_pos=scfg["max_pos"],
                )
                if len(pos) == 0:
                    raise ValueError(f"{sym}: empty position series")

                sig = float(pos.iloc[-1])
                px = float(close_ser.iloc[-1])
                if sig > 0:
                    recs.append({"ticker": sym, "signal": sig, "price": px})
            except Exception as e:
                errs.append(f"{sym}: {e}")

        if not recs:
            return {"status": "error", "message": "No positions computed", "debug": errs}

        df_recs = pd.DataFrame(recs).sort_values("signal", ascending=False)
        top = df_recs.head(pcfg["top_k"])

        w = min(1.0 / len(top), pcfg["max_weight"]) if len(top) > 0 else 0
        weights = {row["ticker"]: w for _, row in top.iterrows()} if w > 0 else {}

        return {
            "status": "ok",
            "recommendations": top.to_dict(orient="records"),
            "portfolio": {"cash": cash, "weights": weights},
            "debug": {"errors": errs},
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
