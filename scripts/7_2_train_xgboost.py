#!/usr/bin/env python3
"""
Step 7.2 — Train XGBoost cylinder-state classifier from corrected CSV data.

Workflow:
    1. Run 7_velocity_check.py --log session.csv  (collect data)
    2. Watch screen recording, fix wrong 'state' values in the CSV manually
    3. Run this script with the corrected CSV to train the model
    4. The saved model can replace the rule-based thresholds in 7_velocity_check.py

Cross-validation holds out whole recording SESSIONS by default (--cv-group
session, needs >=2 input CSVs) rather than small same-state "events" — a
model that shortcuts on absolute pixel position (cx, cy) instead of true
motion can still score well when only individual events are held out,
since other events from the same camera/rig setup stay in the training
set; holding out a whole unseen session tests real generalization, much
closer to what a live run on a new session actually demands.

Usage:
    python3 scripts/7_2_train_xgboost.py session1.csv session2.csv session3.csv
    python3 scripts/7_2_train_xgboost.py data/s1.csv data/s2.csv --window 5
    python3 scripts/7_2_train_xgboost.py session.csv --out models/cylinder_state_xgb.json
    python3 scripts/7_2_train_xgboost.py s1.csv s2.csv --cv-group event   # old behavior, for comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
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

def build_features(df: pd.DataFrame, window: int, group_by: str = "session"
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    For each row build a feature vector using:
      - current frame values for all BASE_COLS
      - lagged values (1..window-1 frames back) for all BASE_COLS
      - first-order deltas (current - lag1) for all BASE_COLS

    Rows that don't have enough history are dropped, as are rows whose
    window would reach back across a session ("_source") boundary.

    group_by controls what cross-validation holds out as one unit:
      "event"   — a contiguous same-state, same-session run. Weak isolation:
                  other events from the SAME recording session (same camera
                  position/lighting/rig) are still in the training set for
                  that fold, so a model that shortcuts on absolute pixel
                  position (cx, cy) rather than true motion can still score
                  well without generalizing to a genuinely new session.
      "session" — a whole CSV file (one recording session). Held-out folds
                  contain sessions the model has NEVER seen any frame of —
                  much closer to what "run live, on a new session" actually
                  demands. Needs >=2 distinct input CSVs to mean anything.

    Returns X (n_samples, n_features), y (n_samples,), groups (n_samples,).
    """
    arr = df[BASE_COLS].values.astype(np.float32)
    labels = df["state"].values
    sources = df["_source"].values
    le = LabelEncoder()
    le.fit(STATES)
    y_all = le.transform(labels)

    if group_by == "session":
        group_id = pd.factorize(df["_source"])[0]
    else:
        group_id = ((df["state"] != df["state"].shift()) |
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
        rows_g.append(group_id[i])

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

def train(X: np.ndarray, y: np.ndarray, groups: np.ndarray, le: LabelEncoder, group_by: str = "session"):
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

    # cap folds at the rarest class's group count, so no fold is forced to
    # hold out 100% of a rare class's groups (still best-effort beyond that —
    # StratifiedGroupKFold approximates stratification when groups are scarce)
    groups_per_class = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, min(groups_per_class)))

    unit = "session" if group_by == "session" else "event"
    print(f"\n── Cross-validation ({n_splits}-fold, grouped by {unit}) ──")
    if n_splits < 5:
        print(f"  [NOTE] Rarest class has only {min(groups_per_class)} distinct {unit}s "
              f"(out of {n_groups} total) — using {n_splits} folds instead of 5, "
              f"and treat this score as a rough signal, not a precise estimate.")
    if group_by == "session" and n_groups < 2:
        print("  [WARNING] Only 1 session in the input data — session-grouped CV "
              "can't hold out a session it still has something to train on. "
              "Pass multiple session CSVs, or use --cv-group event instead.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, groups=groups, cv=cv,
                                 scoring=["accuracy", "f1_macro"], n_jobs=-1)
    acc_scores = cv_results["test_accuracy"]
    f1_scores = cv_results["test_f1_macro"]
    print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"  F1-macro: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Out-of-fold predictions: each row predicted only by a model that never
    # saw it (or its group) during training. Pooling these into one report
    # gives a genuine per-class held-out breakdown — this is the number to
    # trust for "which state is the model actually bad at generalizing on",
    # unlike the training-set report further below.
    print(f"\n── Out-of-fold classification report (pooled across all {unit} "
          f"held-out predictions — this is the genuine generalization estimate "
          f"per class) ──")
    y_pred_oof = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)
    oof_report_str = classification_report(y, y_pred_oof, target_names=le.classes_, zero_division=0)
    print(oof_report_str)
    oof_report_dict = classification_report(y, y_pred_oof, target_names=le.classes_,
                                              zero_division=0, output_dict=True)

    print("── Out-of-fold confusion matrix ──")
    oof_cm = confusion_matrix(y, y_pred_oof)
    header = "        " + "  ".join(f"{s:>11}" for s in le.classes_)
    print(header)
    for i, row in enumerate(oof_cm):
        print(f"{le.classes_[i]:>11}  " + "  ".join(f"{v:>11}" for v in row))

    print("\n── Training on full dataset (for the deployable model) ──")
    model.fit(X, y)

    y_pred = model.predict(X)
    print("\n── Classification report (training set — near-100% is expected and "
          "MEANINGLESS here; the model is scoring itself on data it just "
          "memorized. Trust the out-of-fold report above instead) ──")
    report_str = classification_report(y, y_pred, target_names=le.classes_, zero_division=0)
    print(report_str)
    report_dict = classification_report(y, y_pred, target_names=le.classes_,
                                         zero_division=0, output_dict=True)

    print("── Confusion matrix ──")
    cm = confusion_matrix(y, y_pred)
    header = "        " + "  ".join(f"{s:>11}" for s in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:>11}  " + "  ".join(f"{v:>11}" for v in row))

    cv_metrics = {
        "cv_group_by": group_by,
        "n_splits": n_splits,
        "n_groups": n_groups,
        "cross_validated_accuracy_mean": float(acc_scores.mean()),
        "cross_validated_accuracy_std": float(acc_scores.std()),
        "cross_validated_accuracy_per_fold": [float(s) for s in acc_scores],
        "cross_validated_f1_macro_mean": float(f1_scores.mean()),
        "cross_validated_f1_macro_std": float(f1_scores.std()),
        "cross_validated_f1_macro_per_fold": [float(s) for s in f1_scores],
        "class_labels": list(le.classes_),
        "out_of_fold_classification_report": oof_report_dict,
        "out_of_fold_confusion_matrix": oof_cm.tolist(),
        "training_set_classification_report": report_dict,
        "training_set_confusion_matrix": cm.tolist(),
    }
    return model, cv_metrics


def print_importances(model: XGBClassifier, names: list[str], top_n: int = 20) -> list[dict]:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print(f"\n── Top-{top_n} feature importances ──")
    top = []
    for rank, idx in enumerate(order[:top_n], 1):
        print(f"  {rank:2d}. {names[idx]:<30s} {imp[idx]:.4f}")
        top.append({"feature": names[idx], "importance": float(imp[idx])})
    return top


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", help="Corrected CSV file(s) from 7_velocity_check.py")
    p.add_argument("--window", type=int, default=5,
                   help="Number of past frames used as features (default: 5)")
    p.add_argument("--out", default="models/cylinder_state_xgb.json",
                   help="Where to save the trained model")
    p.add_argument("--cv-group", choices=["session", "event"], default="session",
                   help="What cross-validation holds out per fold: a whole recording "
                        "session (default — needs >=2 input CSVs, tests generalization "
                        "to an unseen session, closest to real live performance) or just "
                        "a contiguous same-state event (weaker isolation, can look "
                        "artificially high if the model shortcuts on absolute pixel "
                        "position instead of true motion)")
    return p.parse_args()


def main():
    args = parse_args()

    print("── Loading CSV files ──")
    frames = [load_csv(p) for p in args.csvs]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(df)}\n")
    summarize_events(df)

    n_sources = df["_source"].nunique()
    cv_group = args.cv_group
    if cv_group == "session" and n_sources < 2:
        print(f"  [NOTE] Only 1 input CSV -> session-grouped CV is meaningless here, "
              f"falling back to --cv-group event.")
        cv_group = "event"

    print("── Building features ──")
    X, y, groups, le = build_features(df, args.window, group_by=cv_group)
    names = feature_names(args.window)
    print(f"  X shape: {X.shape}  (samples × features)")

    model, cv_metrics = train(X, y, groups, le, group_by=cv_group)
    top_importances = print_importances(model, names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save XGBoost model (native format) + label encoder
    model.save_model(str(out_path))
    le_path = out_path.with_suffix(".labels.pkl")
    joblib.dump(le, le_path)

    # save all metrics (this is the file to check for cross-validated
    # accuracy/F1-macro after the fact — nothing above is logged anywhere
    # except this file and your terminal scrollback)
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "csv_files": [str(p) for p in args.csvs],
        "window": args.window,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        **cv_metrics,
        "top_feature_importances": top_importances,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n── Saved ──")
    print(f"  Model:         {out_path}")
    print(f"  Label encoder: {le_path}")
    print(f"  Metrics:       {metrics_path}")
    print(f"\nTo use in real-time, run 7_velocity_check.py with:")
    print(f"  --model {out_path} --window {args.window}")


if __name__ == "__main__":
    main()
