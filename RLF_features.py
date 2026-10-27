"""
Feature engineering and labelling for RLF prediction.

Input: one row per (ue_id, t) at a fixed 1 s cadence, sorted or sortable by
(ue_id, t). All rolling features are BACKWARD-looking (they only use the current
and past samples), so there is no look-ahead leakage in the features. Only the
label looks forward.

Bump FEATURE_VERSION whenever the feature logic changes and log it in MLflow.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = "rlf_features_v1"

# Columns that are identifiers, raw events or labels, never model inputs.
NON_FEATURE_COLS = {"ue_id", "cell_id", "t", "rlf", "ho_cmd", "label"}

ROLL_COLS = ["rsrp", "rsrq", "sinr", "ul_sinr", "cqi", "dl_bler",
             "harq_retx", "ta", "phr", "nb_gap"]
WINDOWS = (5, 30)          # seconds
A3_CAP_S = 600             # cap for "time since last A3 event"


def make_labels(df, horizon_s):
    """label[t] = 1 if an RLF occurs in (t, t + horizon_s] for the same UE.

    Requires df sorted by (ue_id, t) with a contiguous 1 s time index per UE.
    """
    rlf_all = df["rlf"].to_numpy()
    out = np.zeros(len(df), dtype=np.int8)
    for _, idx in df.groupby("ue_id", sort=False).indices.items():
        r = rlf_all[idx]
        # cov[i] = max(r[i : i + horizon_s])
        cov = (pd.Series(r[::-1]).rolling(horizon_s, min_periods=1).max()
                 .to_numpy()[::-1])
        lab = np.zeros_like(r)
        lab[:-1] = cov[1:]          # shift so the window is (t, t + H]
        out[idx] = lab
    return out


def add_features(df):
    df = df.sort_values(["ue_id", "t"]).reset_index(drop=True).copy()

    # Neighbor comparison: how far the best neighbor is above/below the server.
    df["nb_gap"] = df["neighbor_rsrp"] - df["rsrp"]
    df["a3_margin"] = df["nb_gap"] - df["a3_offset"] - df["hysteresis"]
    df["_gap_pos"] = (df["nb_gap"] > 0).astype(float)

    g = df.groupby("ue_id", sort=False)
    feat = {}

    for c in ROLL_COLS:
        for w in WINDOWS:
            roll = g[c].rolling(w, min_periods=1)
            feat[f"{c}_mean_{w}s"] = roll.mean().reset_index(level=0, drop=True)
            feat[f"{c}_std_{w}s"] = roll.std().reset_index(level=0, drop=True)
            feat[f"{c}_min_{w}s"] = roll.min().reset_index(level=0, drop=True)
            # Slope proxy: change over the window divided by its length.
            feat[f"{c}_slope_{w}s"] = (df[c] - g[c].shift(w)) / w

    # Measurement-report / mobility event features.
    for w in WINDOWS:
        feat[f"a3_count_{w}s"] = (g["a3_events"].rolling(w, min_periods=1).sum()
                                  .reset_index(level=0, drop=True))
    feat["ho_count_30s"] = (g["ho_cmd"].rolling(30, min_periods=1).sum()
                            .reset_index(level=0, drop=True))
    feat["gap_pos_frac_30s"] = (g["_gap_pos"].rolling(30, min_periods=1).mean()
                                .reset_index(level=0, drop=True))

    last_a3 = df["t"].where(df["a3_events"] > 0)
    last_a3 = last_a3.groupby(df["ue_id"]).ffill()
    feat["secs_since_a3"] = ((df["t"] - last_a3).fillna(A3_CAP_S)
                             .clip(upper=A3_CAP_S))

    # Timing-advance drift (UE moving away from / towards the cell).
    feat["ta_drift_10s"] = df["ta"] - g["ta"].shift(10)

    out = pd.concat([df.drop(columns=["_gap_pos"]), pd.DataFrame(feat)], axis=1)
    new_cols = list(feat.keys())
    out[new_cols] = out[new_cols].fillna(0.0)
    return out


def build_dataset(df, horizon_s):
    out = add_features(df)
    out["label"] = make_labels(out, horizon_s)
    return out


def feature_columns(df):
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("_")]
