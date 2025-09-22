# frontfin_infer.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
from typing import List, Optional, Dict, Any

BASE = Path("data_cache")
ART = BASE / "artifacts"
FEATURES_PATH = BASE / "features.parquet"

def _resolve_artifacts(family: str | None, horizon_key: str | None) -> Dict[str, Path]:
    """
    Try a few organized paths first, then fall back to your original files.
    """
    candidates = []
    if family and horizon_key:
        candidates += [ART / family / horizon_key, ART / horizon_key / family]
    if family:
        candidates += [ART / family]
    candidates += [ART]

    for d in candidates:
        clf = d / "clf.joblib"
        if not clf.exists():
            # your original artifact name
            clf = d / "clf_logreg.joblib"
        meta = d / "metadata_cls.json"
        if clf.exists() and meta.exists():
            return {"clf": clf, "meta": meta}

    raise FileNotFoundError("Could not find artifacts (clf + metadata_cls.json) under data_cache/artifacts/")

class LoadedModel:
    def __init__(self, clf_path: Path, meta_path: Path, features_path: Path = FEATURES_PATH):
        self.clf = joblib.load(clf_path)
        self.meta = json.loads(meta_path.read_text())
        self.feature_cols = self.meta.get("feature_cols") or self.meta.get("features") or []
        self.df = pd.read_parquet(features_path)

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])

        if "Ticker" not in self.df.columns:
            if "symbol" in self.df.columns:
                self.df = self.df.rename(columns={"symbol": "Ticker"})
            else:
                raise ValueError("Features parquet must contain 'Ticker' column")

    def as_of(self) -> str:
        if "Date" in self.df.columns and not self.df["Date"].empty:
            return str(self.df["Date"].max().date())
        return "today"

    def rank_latest(self, universe: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        df = self.df
        if universe:
            df = df[df["Ticker"].isin(universe)]
            if df.empty:
                return []

        # latest row per ticker
        if "Date" in df.columns:
            latest = df.sort_values(["Ticker", "Date"]).groupby("Ticker", as_index=False).tail(1)
        else:
            latest = df.drop_duplicates(subset=["Ticker"], keep="last")

        X = latest[self.feature_cols]
        proba = self.clf.predict_proba(X)[:, 1]
        latest = latest.assign(prob_up=proba)

        latest = latest.sort_values("prob_up", ascending=False)

        if "Close" not in latest.columns and "close" in latest.columns:
            latest = latest.rename(columns={"close": "Close"})

        return [
            {"ticker": r.Ticker,
             "close": float(getattr(r, "Close", float("nan"))),
             "prob_up": float(r.prob_up)}
            for r in latest.itertuples()
        ]

class ModelRegistry:
    """
    Simple cache so we don't re-load artifacts on every request.
    """
    def __init__(self):
        self.cache: Dict[str, LoadedModel] = {}

    def get(self, family: Optional[str], horizon_key: Optional[str]) -> LoadedModel:
        key = f"{family or 'default'}::{horizon_key or 'default'}"
        if key in self.cache:
            return self.cache[key]
        paths = _resolve_artifacts(family, horizon_key)
        mdl = LoadedModel(paths["clf"], paths["meta"])
        self.cache[key] = mdl
        return mdl
