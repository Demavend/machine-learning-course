from statsmodels.tsa.seasonal import seasonal_decompose
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import inspect
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.models import XGBModel


def decompose_sales(df: pd.DataFrame, item_id: int, store_id: int, period: int = 7, model: str = 'additive'):
    """
    Performs additive or multiplicative decomposition on sales data
    for a specific item and store using statsmodels.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'item', 'store', 'date', 'sales'
    item_id : int
        ID of the item to analyze
    store_id : int
        ID of the store to analyze
    period : int
        Seasonality period (default is 7 for weekly)
    model : str
        Decomposition model: 'additive' or 'multiplicative'

    Returns:
    -------
    result : statsmodels DecomposeResult
        Contains trend, seasonal, resid, observed
    fig : matplotlib.figure.Figure
        The plotted decomposition figure
    """
    subset = df[(df['item'] == item_id) & (df['store'] == store_id)].copy()
    subset = subset.sort_values("date")

    if 'date' in subset.columns:
        if not np.issubdtype(subset['date'].dtype, np.datetime64):
            subset['date'] = pd.to_datetime(subset['date'])
        subset.set_index('date', inplace=True)
    elif subset.index.name != 'date':
        raise ValueError("Expected a 'date' column or index named 'date'")

    result = seasonal_decompose(subset['sales'], model=model, period=period)

    fig = result.plot()
    fig.suptitle(f'Decomposition for Item {item_id}, Store {store_id} ({model.capitalize()})', fontsize=14)
    fig.tight_layout()

    return result, fig


# ---- Feature engineering (calendar + safe aggregates from TRAIN only)

def _infer_name(candidates: List[str], columns: List[str]) -> Optional[str]:
    """Case-insensitive column name inference."""
    low = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def standardize_schema(
    df: pd.DataFrame,
    cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Ensure canonical column names: date, store, item, sales.
    - If `cols` is provided, rename using it (keys: date/store/item/sales or 'target' -> sales).
    - Otherwise infer by common aliases.
    - If date column is missing but index is datetime-like, move index to 'date'.
    """
    out = df.copy()

    # 1) explicit mapping
    if cols:
        rename_map = {}
        for canon, given in cols.items():
            canon_name = "sales" if canon.lower() in ("target", "sales") else canon
            if given in out.columns:
                rename_map[given] = canon_name
        if rename_map:
            out = out.rename(columns=rename_map)

    # 2) inference for missing ones (case-insensitive)
    colnames = list(out.columns)

    if "date" not in out.columns:
        date_alias = _infer_name(["date", "Date", "ds", "timestamp", "time"], colnames)
        if date_alias:
            out = out.rename(columns={date_alias: "date"})
        else:
            # maybe datetime-like index
            idx = out.index
            if getattr(idx, "inferred_type", "") in ("datetime64", "datetime", "date") or \
               (idx.name and str(idx.name).lower() in ("date", "ds", "timestamp", "time")):
                out = out.reset_index().rename(columns={idx.name or "index": "date"})
            else:
                raise KeyError(
                    "Could not find a date column. "
                    "Provide `cols={'date': '<your_date_col>', 'store': '...', 'item': '...', 'sales': '...'}"
                    " or reset a datetime index into a column."
                )

    if "store" not in out.columns:
        store_alias = _infer_name(["store", "Store", "shop", "branch", "location"], colnames)
        if store_alias: out = out.rename(columns={store_alias: "store"})
        else: raise KeyError("Missing 'store' column (use `cols={'store': '...'}`)")

    if "item" not in out.columns:
        item_alias = _infer_name(["item", "Item", "product", "sku", "sku_id", "product_id"], colnames)
        if item_alias: out = out.rename(columns={item_alias: "item"})
        else: raise KeyError("Missing 'item' column (use `cols={'item': '...'}`)")

    if "sales" not in out.columns:
        sales_alias = _infer_name(["sales", "Sales", "target", "y", "value", "qty", "quantity"], colnames)
        if sales_alias: out = out.rename(columns={sales_alias: "sales"})
        else: raise KeyError("Missing 'sales' column (use `cols={'sales': '...'} or {'target': '...'}`)")

    # Parse dates (Pandas now uses strict parser by default)
    out["date"] = pd.to_datetime(out["date"])
    if out["date"].isna().all():
        raise ValueError("Failed to parse 'date' to datetime. Check formats.")

    return out


# =========================
# Date features & resampling
# =========================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (+ simple cyclic encodings)."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    iso = out["date"].dt.isocalendar()
    out["year"]      = out["date"].dt.year
    out["month"]     = out["date"].dt.month
    out["day"]       = out["date"].dt.day
    out["week"]      = iso.week.astype(int)
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["quarter"]   = out["date"].dt.quarter

    out["is_month_start"]   = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"]     = out["date"].dt.is_month_end.astype(int)
    out["is_quarter_start"] = out["date"].dt.is_quarter_start.astype(int)
    out["is_quarter_end"]   = out["date"].dt.is_quarter_end.astype(int)
    out["is_year_start"]    = out["date"].dt.is_year_start.astype(int)
    out["is_year_end"]      = out["date"].dt.is_year_end.astype(int)

    # Cyclic encodings
    for col, period in [("month", 12), ("dayofweek", 7), ("day", 31)]:
        out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


def _norm_freq(freq: str) -> str:
    """Normalize to the user's desired display freq; prefer 'ME' over 'M'."""
    f = (freq or "D").upper()
    return f


def _resample_df(full_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample per (item, store) to requested frequency (sum)."""
    f = _norm_freq(freq)

    df = full_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    g = (
        df.groupby(["item", "store", pd.Grouper(key="date", freq=f)], as_index=False)["sales"]
          .sum()
          .sort_values("date")
    )
    return g


# =========================
# Train-only aggregates
# =========================

def compute_train_aggregates(full_df: pd.DataFrame, split_date: pd.Timestamp, freq: str = "D") -> Dict[str, Any]:
    """
    Build train-only aggregation lookups to avoid leakage.
    Aggregations adapt to the target frequency.
    """
    f = _norm_freq(freq)
    df = _resample_df(full_df, f)
    df = add_calendar_features(df)

    train_df = df[df["date"] < split_date].copy()
    lookups: Dict[str, Any] = {}

    # Period-aware mean by (item, store, period_key)
    if f == "D":
        key_name, avg_name = "dayofweek", "daily_avg"
    elif f.startswith("W"):
        key_name, avg_name = "week", "weekly_avg"
    else:  # monthly & others
        key_name, avg_name = "month", "monthly_avg"

    lookups[("item", "store", key_name, avg_name)] = (
        train_df.groupby(["item", "store", key_name])["sales"].mean().rename(avg_name)
    )

    # Month-of-year mean (keeps monthly seasonality)
    lookups[("item", "store", "month", "monthly_avg")] = (
        train_df.groupby(["item", "store", "month"])["sales"].mean().rename("monthly_avg")
    )

    # Monthly totals across all stores/items
    lookups[("item", "month", "item_month_sum")] = (
        train_df.groupby(["item", "month"])["sales"].sum().rename("item_month_sum")
    )
    lookups[("store", "month", "store_month_sum")] = (
        train_df.groupby(["store", "month"])["sales"].sum().rename("store_month_sum")
    )

    lookups["global_mean"] = float(train_df["sales"].mean()) if len(train_df) else 0.0
    lookups["month_sum_mean"] = float(train_df.groupby("month")["sales"].sum().mean()) if len(train_df) else 0.0
    lookups["_avg_name"] = avg_name
    lookups["_freq"] = f
    return lookups


def merge_aggregates(sdf: pd.DataFrame, lookups: Dict[str, Any]) -> pd.DataFrame:
    """Merge train-only aggregates into the single-series frame (works for D/W/M)."""
    out = sdf.copy()

    for key in [
        ("item", "store", "dayofweek", "daily_avg"),
        ("item", "store", "week", "weekly_avg"),
        ("item", "store", "month", "monthly_avg"),
        ("item", "month", "item_month_sum"),
        ("store", "month", "store_month_sum"),
    ]:
        if key in lookups:
            col = key[-1]
            out = out.merge(lookups[key].reset_index(), on=list(key[:-1]), how="left")

    gm = lookups.get("global_mean", 0.0)
    msm = lookups.get("month_sum_mean", 0.0)
    for c in ["daily_avg", "weekly_avg", "monthly_avg"]:
        if c in out.columns:
            out[c] = out[c].fillna(gm)
    for c in ["item_month_sum", "store_month_sum"]:
        if c in out.columns:
            out[c] = out[c].fillna(msm)
    return out


# =========================
# Bundle (target + covariates)
# =========================

@dataclass
class SeriesBundle:
    """Container holding target and covariates for a single item×store series."""
    series: TimeSeries
    past_cov: Optional[TimeSeries]
    future_cov: TimeSeries
    train: TimeSeries
    val: TimeSeries
    past_train: Optional[TimeSeries]
    future_train: TimeSeries
    split_date: pd.Timestamp


def build_series_bundle(
    df: pd.DataFrame,
    item_id: int,
    store_id: int,
    freq: str,
    split_date: pd.Timestamp,
    cols: Optional[Dict[str, str]] = None,
) -> SeriesBundle:
    """Prepare TimeSeries target & covariates for a single (item, store), resampled to `freq`."""
    df = standardize_schema(df, cols=cols)

    f = _norm_freq(freq)
    sub = df.loc[(df["item"] == item_id) & (df["store"] == store_id), ["date", "sales"]].copy()
    sub["date"] = pd.to_datetime(sub["date"])

    # Resample to target freq (sum) and ensure continuity
    s = (
        sub.set_index("date")["sales"]
           .resample(f).sum()
           .asfreq(f)
           .fillna(0.0)
           .reset_index()
           .rename(columns={"sales": "sales"})
    )
    s["item"] = item_id
    s["store"] = store_id

    # Features + train-only aggregates (computed at SAME freq)
    s = add_calendar_features(s)
    lookups = compute_train_aggregates(df, split_date, freq=f)
    s = merge_aggregates(s, lookups)

    # Build series
    y = TimeSeries.from_dataframe(s, time_col="date", value_cols="sales", freq=f)

    candidate_past = ["daily_avg", "weekly_avg", "monthly_avg", "item_month_sum", "store_month_sum"]
    past_cols = [c for c in candidate_past if c in s.columns]
    past_cov = TimeSeries.from_dataframe(s, "date", past_cols, freq=f) if past_cols else None

    fut_cols = [
        "year","month","day","week","dayofweek","dayofyear","quarter",
        "is_month_start","is_month_end","is_quarter_start","is_quarter_end","is_year_start","is_year_end",
        "month_sin","month_cos","dayofweek_sin","dayofweek_cos","day_sin","day_cos"
    ]
    future_cov = TimeSeries.from_dataframe(s, "date", [c for c in fut_cols if c in s.columns], freq=f)

    # Split
    train, val = y.split_before(split_date)
    past_train   = past_cov.slice_intersect(train) if past_cov is not None else None
    future_train = future_cov.slice_intersect(train)

    return SeriesBundle(
        series=y, past_cov=past_cov, future_cov=future_cov,
        train=train, val=val, past_train=past_train, future_train=future_train,
        split_date=split_date
    )


# =========================
# Training & evaluation (MAPE-only)
# =========================

@dataclass
class FitResult:
    name: str
    mape: float
    pred_scaled: TimeSeries
    val_scaled: TimeSeries
    pred: TimeSeries
    val: TimeSeries


def fit_predict_mape(
    model: Any,
    bundle: SeriesBundle,
    scale: bool = True
) -> FitResult:
    """
    Train a Darts model on train split and evaluate on validation using MAPE.
    Works for models with/without covariates (auto-detects supported args).
    """
    # Scalers
    scaler_y = Scaler() if scale else None
    scaler_p = Scaler() if (scale and bundle.past_train is not None) else None
    scaler_f = Scaler() if scale else None

    train_s = scaler_y.fit_transform(bundle.train) if scaler_y else bundle.train
    val_s   = scaler_y.transform(bundle.val) if scaler_y else bundle.val

    past_train_s = None
    past_cov_s   = None
    if bundle.past_train is not None:
        past_train_s = scaler_p.fit_transform(bundle.past_train) if scaler_p else bundle.past_train
        past_cov_s   = scaler_p.transform(bundle.past_cov) if (scaler_p and bundle.past_cov is not None) else bundle.past_cov

    future_train_s = scaler_f.fit_transform(bundle.future_train) if scaler_f else bundle.future_train
    future_cov_s   = scaler_f.transform(bundle.future_cov) if scaler_f else bundle.future_cov

    # Prepare kwargs by inspecting model's signature
    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs = {}
    if "past_covariates" in fit_sig and past_train_s is not None:
        fit_kwargs["past_covariates"] = past_train_s
    if "future_covariates" in fit_sig and future_train_s is not None:
        fit_kwargs["future_covariates"] = future_train_s

    model.fit(train_s, **fit_kwargs)

    pred_sig = inspect.signature(model.predict).parameters
    pred_kwargs = {"n": len(val_s), "show_warnings": False}
    if "past_covariates" in pred_sig and past_cov_s is not None:
        pred_kwargs["past_covariates"] = past_cov_s
    if "future_covariates" in pred_sig and future_cov_s is not None:
        pred_kwargs["future_covariates"] = future_cov_s

    pred_s = model.predict(**pred_kwargs)

    # Inverse scaling for readability
    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred_s)
        val = scaler_y.inverse_transform(val_s)
    else:
        pred, val = pred_s, val_s

    score = float(mape(val, pred))
    name = getattr(model, "model_name", model.__class__.__name__)
    return FitResult(name=name, mape=score,
                     pred_scaled=pred_s, val_scaled=val_s,
                     pred=pred, val=val)


def evaluate_models(
    models: Dict[str, Any],
    bundle: SeriesBundle,
    scale: bool = True
) -> pd.DataFrame:
    """Train & evaluate multiple models on the same bundle. Returns DataFrame with MAPE."""
    rows = []
    for name, model in models.items():
        if not hasattr(model, "model_name"):
            try:
                model.model_name = name
            except Exception:
                pass
        res = fit_predict_mape(model, bundle, scale=scale)
        rows.append({"model": res.name, "mape": res.mape})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


# =========================
# Freq-aware helpers
# =========================

def make_xgb_by_freq(freqs: List[str]) -> Dict[str, XGBModel]:
    """Create an XGBModel per frequency with sensible lags."""
    models: Dict[str, XGBModel] = {}
    for f in freqs:
        nf = _norm_freq(f)
        if nf == "ME":            # monthly
            lags = [-1, -2, -3, -6, -12]
        elif nf.startswith("W"): # weekly
            lags = [-1, -2, -4, -8, -12]
        else:                    # daily
            lags = [-1, -2, -7, -14, -28]

        m = XGBModel(
            lags=lags,
            lags_past_covariates=[lag for lag in lags if lag < 0],
            lags_future_covariates=[0],
            output_chunk_length=1,
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=0,
        )
        m.model_name = f"XGB({f})"
        models[f] = m
    return models


def build_bundles_by_freq(
    df: pd.DataFrame,
    item_id: int,
    store_id: int,
    split_date: pd.Timestamp,
    freqs: List[str]
) -> Dict[str, SeriesBundle]:
    """Build a SeriesBundle for each frequency."""
    return {
        f: build_series_bundle(df, item_id=item_id, store_id=store_id, freq=f, split_date=split_date)
        for f in freqs
    }


def evaluate_by_freq(
    models_by_freq: Dict[str, Any],
    bundles_by_freq: Dict[str, SeriesBundle],
    scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, FitResult]]:
    """Train & evaluate models per frequency. Returns scoreboard + detailed results."""
    rows, results = [], {}
    for f, model in models_by_freq.items():
        res = fit_predict_mape(model, bundles_by_freq[f], scale=scale)
        results[f] = res
        rows.append({"freq": f, "mape": res.mape})
    tbl = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    return tbl, results
