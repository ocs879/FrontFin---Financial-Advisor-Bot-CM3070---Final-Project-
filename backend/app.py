# app.py
from __future__ import annotations
from typing import Optional, List, Dict
from typing_extensions import Annotated
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import timedelta, date
from frontfin_infer import ModelRegistry
import pandas as pd

app = FastAPI(title="FrontFin API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REG = ModelRegistry()

# ---------- Request / Response Schemas ----------
class RecommendRequest(BaseModel):
    age: Optional[str] = None          # "18-25", "26-35", "36-50", "51-65", "65+"
    traderType: Optional[str] = None   # Speedster / Intuitive / Conservative / Strategist / Risk / Adaptive  
    stockOpt: Optional[str] = None     # AAPL / MSFT / GOOGL / TSLA  
    investment_amount: float = Field(..., gt=0)
    time_horizon: Optional[str] = None # "short" | "medium" | "long"  
    risk_tolerance: Optional[str] = None # "Technical" | "Fundamental" | "Sentiment" | "Intuitive"  
    top_k: int = Field(5, ge=1, le=20)
    universe: Optional[Annotated[List[str], Field(min_length=1)]] = None  # optional explicit basket

class Pick(BaseModel):
    ticker: str
    close: float
    prob_up: float
    weight: float
    shares: int

class RecommendResponse(BaseModel):
    as_of: str
    horizon_days: int
    top_k: int
    picks: List[Pick]
    cash_left: float
    notes: str
    policy_used: Dict[str, str | float | int]

# ---------- Policy Mappers (every input matters) ----------
def map_horizon_to_key_days(h: Optional[str]) -> tuple[str, int]:
    """
    Map UI selection to artifact key + horizon days.
    We still fall back to your current 21-day model if others don't exist.
    """
    if not h: return ("h21", 21)
    h = h.lower()
    if "short" in h:   return ("h5", 5)
    if "medium" in h:  return ("h21", 21)
    if "long" in h:    return ("h63", 63)
    return ("h21", 21)

def map_strategy_to_family(s: Optional[str]) -> str:
    if not s: return "technical"
    s = s.lower()
    if "tech" in s:       return "technical"
    if "fund" in s:       return "fundamental"
    if "sent" in s:       return "sentiment"
    # "Intuitive" falls back to current technical model, but we still tag it
    return "technical"

def map_age_to_baseline(age: Optional[str]) -> dict:
    """
    Age → baseline risk (max per name + cash buffer).
    """
    if not age: return {"max_weight": 0.25, "cash_buffer": 0.10}
    a = str(age)
    if "18-25" in a: return {"max_weight": 0.35, "cash_buffer": 0.00}
    if "26-35" in a: return {"max_weight": 0.30, "cash_buffer": 0.05}
    if "36-50" in a: return {"max_weight": 0.25, "cash_buffer": 0.10}
    if "51-65" in a: return {"max_weight": 0.20, "cash_buffer": 0.15}
    if "65+"   in a: return {"max_weight": 0.15, "cash_buffer": 0.20}
    # numeric fallback
    try:
        n = int(a)
        if n <= 25: return {"max_weight": 0.35, "cash_buffer": 0.00}
        if n <= 35: return {"max_weight": 0.30, "cash_buffer": 0.05}
        if n <= 50: return {"max_weight": 0.25, "cash_buffer": 0.10}
        if n <= 65: return {"max_weight": 0.20, "cash_buffer": 0.15}
        return {"max_weight": 0.15, "cash_buffer": 0.20}
    except:  # noqa
        return {"max_weight": 0.25, "cash_buffer": 0.10}

def map_trader_type(tt: Optional[str], horizon_days: int) -> dict:
    """
    Trader type → probability threshold + aggressiveness knobs.
    """
    base = {"prob_threshold": 0.50, "top_k_bonus": 0}
    if not tt: return base
    t = tt.lower()
    if "conserv" in t:
        return {"prob_threshold": 0.55, "top_k_bonus": 0}
    if "strateg" in t:
        return {"prob_threshold": 0.52, "top_k_bonus": 0}
    if "speed" in t:
        return {"prob_threshold": 0.48, "top_k_bonus": 1}
    if "risk" in t:
        return {"prob_threshold": 0.45, "top_k_bonus": 1}
    if "adaptive" in t:
        # adjust with horizon: shorter horizon can be looser
        return {"prob_threshold": 0.48 if horizon_days <= 21 else 0.52, "top_k_bonus": 0}
    # intuitive → mild filter
    return {"prob_threshold": 0.50, "top_k_bonus": 0}

# def build_universe(stockOpt: Optional[str], explicit: Optional[List[str]]) -> Optional[List[str]]:
#     if explicit: return explicit
#     if stockOpt: return [stockOpt]
#     return None  # use model's full universe

def build_universe(stock_opt: Optional[str], explicit: Optional[List[str]]):
    """
    If user passes an explicit universe, use that.
    Otherwise, use all tickers in the features file, with the user's chosen
    stock (if any) placed first but NOT the only one.
    """
    if explicit:
        # user provided a custom list; keep as-is
        return explicit

    # derive model universe from features parquet
    mdl = REG.get(family=None, horizon_key=None)  # any loaded model gives us df
    all_tickers = (
        mdl.df["Ticker"].dropna().astype(str).unique().tolist()
        if "Ticker" in mdl.df.columns else []
    )

    if not all_tickers:
        return None  # fallback: let ranker use its default

    if stock_opt and stock_opt in all_tickers:
        rest = [t for t in all_tickers if t != stock_opt]
        return [stock_opt] + rest  # preferred first, but keep others
    return all_tickers

# ---------- Core helpers ----------
def allocate(amount: float, ranked: List[dict], max_weight: float, cash_buffer: float, k: int):
    """
    Equal-weight up to max_weight; normalize; buy whole shares; return picks + cash_left.
    """
    if amount <= 0:
        return [], 0.0
    usable = max(amount * (1.0 - cash_buffer), 0.0)
    k = max(min(k, len(ranked)), 1)

    # equal weight, capped
    base = min(1.0 / k, max_weight)
    weights = [base] * k
    s = sum(weights)
    weights = [w / s for w in weights]  # renormalize

    picks = []
    cash_left = amount  # we'll compute spent
    for w, r in zip(weights, ranked[:k]):
        price = float(r.get("close") or 0.0)
        alloc = usable * w
        shares = int(alloc // price) if price > 0 else 0
        spent = shares * price
        cash_left -= spent
        actual_weight = (spent / amount) if amount > 0 else 0.0
        picks.append({
            "ticker": r["ticker"],
            "close": price,
            "prob_up": float(r["prob_up"]),
            "weight": actual_weight,
            "shares": shares
        })

    # ensure at least 1 share if possible
    if all(p["shares"] == 0 for p in picks):
        best = ranked[0]
        price = float(best.get("close") or 0.0)
        if price > 0 and amount >= price:
            shares = int(amount // price)
            picks = [{
                "ticker": best["ticker"], "close": price,
                "prob_up": float(best["prob_up"]), "weight": (shares*price)/amount,
                "shares": shares
            }]
            cash_left = amount - shares*price
    return picks, round(max(cash_left, 0.0), 2)

# ---------- Routes ----------
@app.get("/")
def root():
    return {"msg":"FrontFin API is running. Try /api/health or /docs."}

@app.get("/api/health")
def health():
    return {"status":"ok"}

@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # 1) Determine horizon + model family
    horizon_key, horizon_days = map_horizon_to_key_days(req.time_horizon)
    family = map_strategy_to_family(req.risk_tolerance)

    # 2) Resolve risk baseline from age
    base = map_age_to_baseline(req.age)

    # 3) Tune aggressiveness from trader type
    style = map_trader_type(req.traderType, horizon_days)

    # 4) Compute top_k (traderType can add +1 to k, but cap at request)
    k = max(1, min(req.top_k + style["top_k_bonus"], 20))

    # 5) Load model & rank
    mdl = REG.get(family=family, horizon_key=horizon_key)
    uni = build_universe(req.stockOpt, req.universe)
    ranked = mdl.rank_latest(universe=uni)

    if not ranked:
        raise HTTPException(400, "No candidates found for the selected universe/features.")

    # 6) Take top-k ranked candidates (no display filter) and flag threshold
    top_list = ranked[:k]
    for r in top_list:
        r["meets_threshold"] = r["prob_up"] >= style["prob_threshold"]

    # Find the preferred ticker in the top list (if user selected one)
    preferred = None
    if req.stockOpt:
        want = str(req.stockOpt).upper()
        for r in top_list:
            if str(r["ticker"]).upper() == want:
                preferred = r
                break

    # Allocation universe = those that meet threshold...
    alloc_universe = [r for r in top_list if r["meets_threshold"]]

    # ...but always include the preferred ticker even if below threshold
    if preferred and preferred not in alloc_universe:
        alloc_universe = [preferred] + alloc_universe

    # Fallback: never empty (take #1 if everything is below threshold and no preferred)
    if not alloc_universe:
        alloc_universe = top_list[:1]

    # 7) Allocate with age-based policy
    allocated, cash_left = allocate(
        amount=req.investment_amount,
        ranked=alloc_universe,
        max_weight=base["max_weight"],
        cash_buffer=base["cash_buffer"],
        k=min(k, len(alloc_universe))
    )

    # 7.1 Ensure preferred gets at least 1 share if possible
    # Build quick lookup
    by_ticker = {a["ticker"]: a for a in allocated}
    if preferred:
        tkr = preferred["ticker"]
        got = by_ticker.get(tkr)
        # Only attempt if we didn't get any shares and we can afford one share
        if (not got or got.get("shares", 0) == 0):
            price = float(preferred.get("close") or 0.0)
            if price > 0.0:
                # Respect cash buffer: only spend from (cash_left - buffer)
                total_amt = float(req.investment_amount or 0.0)
                buffer_amt = total_amt * float(base["cash_buffer"])
                spendable = max(0.0, cash_left - buffer_amt)

                # Respect max single-name weight
                max_name_amt = total_amt * float(base["max_weight"])
                already_alloc = sum((a["weight"] * total_amt) for a in allocated)
                # remaining allowable per name (simple check; we only try 1 share)
                if price <= spendable and price <= max_name_amt:
                    # assign 1 share to preferred
                    if got:
                        got["shares"] = 1
                        got["weight"] = price / total_amt if total_amt > 0 else 0.0
                    else:
                        # create a new allocation row
                        got = {
                            "ticker": tkr,
                            "close": price,
                            "prob_up": float(preferred["prob_up"]),
                            "weight": (price / total_amt) if total_amt > 0 else 0.0,
                            "shares": 1
                        }
                        allocated.append(got)
                        by_ticker[tkr] = got
                    cash_left = max(0.0, cash_left - price)

    # 8) Merge: return all top_k (allocated names with weights/shares; others 0)
    by_ticker = {a["ticker"]: a for a in allocated}  # rebuild after potential 1-share tweak
    merged_picks = []
    for r in top_list:
        a = by_ticker.get(r["ticker"])
        if a:
            merged_picks.append({
                "ticker": a["ticker"],
                "close": a["close"],
                "prob_up": a.get("prob_up", float(r["prob_up"])),
                "weight": a["weight"],
                "shares": a["shares"],
                "meets_threshold": r["meets_threshold"] or (preferred and r["ticker"] == preferred["ticker"])
            })
        else:
            merged_picks.append({
                "ticker": r["ticker"],
                "close": float(r.get("close") or 0.0),
                "prob_up": float(r["prob_up"]),
                "weight": 0.0,
                "shares": 0,
                "meets_threshold": r["meets_threshold"]
            })


    policy_used = {
        "family": family,
        "horizon_key": horizon_key,
        "horizon_days": horizon_days,
        "prob_threshold": style["prob_threshold"],
        "top_k_requested": req.top_k,
        "top_k_effective": k,
        "max_weight": base["max_weight"],
        "cash_buffer": base["cash_buffer"],
        "universe": ", ".join(uni) if uni else "model_default"
    }

    return {
        "as_of": mdl.as_of(),
        "horizon_days": horizon_days,
        "top_k": len(merged_picks),
        "picks": merged_picks,        # <-- now includes non-allocated names with 0 weight/shares
        "cash_left": cash_left,
        "notes": "Experimental—educational use only.",
        "policy_used": policy_used
    }

@app.get("/api/history")
def history(ticker: str = Query(..., min_length=1), days: int = 90):
    """
    Return simple daily close history for the given ticker from the same features parquet.
    """
    mdl = REG.get(family=None, horizon_key=None)
    df = mdl.df
    print("DEBUG columns:", df.columns.tolist())
    if "Date" not in df.columns:
        raise HTTPException(400, "Features do not include Date column.")

    # Normalize column name for close price
    if "Close" in df.columns:
        close_col = "Close"
    elif "close" in df.columns:
        close_col = "close"
    else:
        raise HTTPException(400, "Features do not include a Close/close column.")

    sub = df[df["Ticker"] == ticker].sort_values("Date")
    if sub.empty:
        raise HTTPException(404, f"No history for {ticker}.")

    # last N days
    end = sub["Date"].max()
    start = end - pd.Timedelta(days=days+10)
    sub = sub[sub["Date"] >= start]

    return {
        "ticker": ticker,
        "as_of": str(end.date()),
        "dates": [str(d.date()) for d in sub["Date"]],
        "closes": [float(x) for x in sub[close_col]],
    }
