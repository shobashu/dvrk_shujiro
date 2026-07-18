#!/usr/bin/env python3
"""
Step 7.2 — Train XGBoost cylinder-state classifier from corrected CSV data.

Workflow:
    1. Run 7_velocity_check.py --log session.csv  (collect data)
    2. Watch screen recording, fix wrong 'state' values in the CSV manually
    3. Run this script with the corrected CSV to train the model
    4. The saved model can replace the rule-based thresholds in 7_velocity_check.py

Usage:
    python3 scripts/7_2_train_xgboost.py session_corrected.csv
    python3 scripts/7_2_train_xgboost.py data/s1.csv data/s2.csv --window 5
    python3 scripts/7_2_train_xgboost.py session.csv --out models/cylinder_state_xgb.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Features present in each CSV row (from 7_velocity_check.py)
BASE_COLS = [
    "vel_px_s",
    "cx", "cy",
    "psm1_x", "psm1_y", "psm1_z",
    "psm1_qx", "psm1_qy", "psm1_qz", "psm1_qw",
]

STATES = ["STATIONARY", "HELD", "DROPPED", "LOST"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV produced by 7_velocity_check.py, skipping comment lines."""
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    missing = [c for c in ["state"] + BASE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}\n"
                 "Make sure you ran 7_velocity_check.py with PSM1 data and --log.")
    df["state"] = df["state"].str.strip()
    df = df[df["state"].isin(STATES)].copy()
    # drop rows where PSM1 was not yet received (all PSM1 cols are NaN or empty)
    psm1_cols = [c for c in BASE_COLS if c.startswith("psm1_")]
    df = df.dropna(subset=psm1_cols)
    df = df.reset_index(drop=True)
    df["_source"] = Path(path).name  # tags rows so windows/events never cross session boundaries
    print(f"  {path}: {len(df)} rows after cleaning")
    return df


def summarize_events(df: pd.DataFrame) -> None:
    """Print row counts vs. distinct contiguous-event counts per state.

    Row count alone is misleading for a state machine: a rare state that only
    ever occurs in a couple of long segments will still rack up a big row
    count, but the model has really only seen a couple of independent
    examples of what that state looks like.
    """
    seg = ((df["state"] != df["state"].shift()) |
           (df["_source"] != df["_source"].shift())).cumsum()
    n_events = df.groupby(seg)["state"].first().value_counts()
    n_rows = df["state"].value_counts()
    print(f"  {'state':<11}{'rows':>8}{'events':>8}")
    for s in STATES:
        print(f"  {s:<11}{n_rows.get(s, 0):>8}{n_events.get(s, 0):>8}")
    print()


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    For each row build a feature vector using:
      - current frame values for all BASE_COLS
      - lagged values (1..window-1 frames back) for all BASE_COLS
      - first-order deltas (current - lag1) for all BASE_COLS

    Rows that don't have enough history are dropped, as are rows whose
    window would reach back across a session ("_source") boundary.

    Also assigns each row a group id — a contiguous same-state, same-session
    "event" — so that cross-validation can hold out whole events instead of
    individual (near-duplicate) frames. See summarize_events().

    Returns X (n_samples, n_features), y (n_samples,), groups (n_samples,).
    """
    arr = df[BASE_COLS].values.astype(np.float32)
    labels = df["state"].values
    sources = df["_source"].values
    le = LabelEncoder()
    le.fit(STATES)
    y_all = le.transform(labels)

    event_id = ((df["state"] != df["state"].shift()) |
                (df["_source"] != df["_source"].shift())).cumsum().values

    rows_X, rows_y, rows_g = [], [], []
    for i in range(window - 1, len(arr)):
        if sources[i - window + 1] != sources[i]:
            continue  # window would straddle two different recording sessions
        window_data = arr[i - window + 1: i + 1]   # shape (window, n_base)
        # flatten: current frame first, then lag1, lag2, ...
        current = window_data[-1]
        lags    = window_data[:-1][::-1].flatten()   # lag1 first
        delta   = current - window_data[-2] if window >= 2 else np.zeros_like(current)
        feat    = np.concatenate([current, lags, delta])
        rows_X.append(feat)
        rows_y.append(y_all[i])
        rows_g.append(event_id[i])

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    groups = np.array(rows_g, dtype=np.int64)
    return X, y, groups, le


def feature_names(window: int) -> list[str]:
    names = [f"{c}" for c in BASE_COLS]
    for lag in range(1, window):
        names += [f"{c}_lag{lag}" for c in BASE_COLS]
    names += [f"{c}_delta" for c in BASE_COLS]
    return names


# ── Training ──────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, groups: np.ndarray, le: LabelEncoder):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    # cap folds at the rarest class's event count, so no fold is forced to
    # hold out 100% of a rare class's events (still best-effort beyond that —
    # StratifiedGroupKFold approximates stratification when groups are scarce)
    events_per_class = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, min(events_per_class)))

    print(f"\n── Cross-validation ({n_splits}-fold, grouped by event) ──")
    if n_splits < 5:
        print(f"  [NOTE] Rarest class has only {min(events_per_class)} distinct events "
              f"(out of {n_groups} total) — using {n_splits} folds instead of 5, "
              f"and treat this score as a rough signal, not a precise estimate.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")

    print("\n── Training on full dataset ──")
    model.fit(X, y)

    y_pred = model.predict(X)
    print("\n── Classification report (training set — optimistic, not a generalization "
          "estimate; trust the grouped CV score above instead) ──")
    print(classification_report(y, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    print("── Confusion matrix ──")
    cm = confusion_matrix(y, y_pred)
    header = "        " + "  ".join(f"{s:>11}" for s in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:>11}  " + "  ".join(f"{v:>11}" for v in row))

    return model


def print_importances(model: XGBClassifier, names: list[str], top_n: int = 20):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print(f"\n── Top-{top_n} feature importances ──")
    for rank, idx in enumerate(order[:top_n], 1):
        print(f"  {rank:2d}. {names[idx]:<30s} {imp[idx]:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", help="Corrected CSV file(s) from 7_velocity_check.py")
    p.add_argument("--window", type=int, default=5,
                   help="Number of past frames used as features (default: 5)")
    p.add_argument("--out", default="models/cylinder_state_xgb.json",
                   help="Where to save the trained model")
    return p.parse_args()


def main():
    args = parse_args()

    print("── Loading CSV files ──")
    frames = [load_csv(p) for p in args.csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(df)}\n")
    summarize_events(df)

    print("── Building features ──")
    X, y, groups, le = build_features(df, args.window)
    names = feature_names(args.window)
    print(f"  X shape: {X.shape}  (samples × features)")

    model = train(X, y, groups, le)
    print_importances(model, names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save XGBoost model (native format) + label encoder
    model.save_model(str(out_path))
    le_path = out_path.with_suffix(".labels.pkl")
    joblib.dump(le, le_path)

    print(f"\n── Saved ──")
    print(f"  Model:         {out_path}")
    print(f"  Label encoder: {le_path}")
    print(f"\nTo use in real-time, run 7_velocity_check.py with:")
    print(f"  --model {out_path} --window {args.window}")


if __name__ == "__main__":
    main()
