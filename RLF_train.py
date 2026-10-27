"""
Compare model families for RLF prediction with MLflow.

Usage:
    python train.py                                   # synthetic data
    python train.py --horizon 30 --n-ues 80
    python train.py --csv my_kpm_export.csv           # real data, same columns
    mlflow ui --backend-store-uri sqlite:///mlflow.db # browse the runs

Every family sees identical time-based folds and identical metrics, so
"do we need deep learning?" is answered by the numbers, not by opinion.
"""
import argparse
import copy
import inspect
import time

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from RLF_features import FEATURE_VERSION, build_dataset, feature_columns
from RLF_synthetic_data import generate_dataset

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
from torch import nn

SEED = 50
TARGET_FPR = 0.02

BASELINE_COLS = ["rsrp_slope_5s", "sinr_slope_5s", "nb_gap", "dl_bler_mean_5s",
                 "harq_retx_mean_5s", "ta_slope_30s"]


# Raw per-second channels fed to the GRU (it learns its own temporal features).
# prb_util, connected_ues, ttt_ms, hour_of_day are left out: no failure signal.
SEQ_COLS = ["rsrp", "rsrq", "sinr", "ul_sinr", "cqi", "dl_bler", "harq_retx",
            "ta", "phr", "nb_gap", "a3_margin", "a3_events", "ho_cmd"]
 
 
# NOTE on the model interface used by run_family:
#   fit(X, y, idx=None)  and  score(X, idx=None)
# X is the engineered-feature matrix of the selected rows, `idx` are the row
# positions of those rows in the full dataframe. Tabular models ignore `idx`;
# the GRU uses it to look up each row's preceding seconds.
 
# ----------------------------------------------------------------- models ---
class RobustZ:
    """Unsupervised baseline: max |robust z-score| over a few KPI features."""
    name = "robust_z_baseline"
    params = {"cols": ",".join(BASELINE_COLS)}

    def __init__(self, feature_names):
        self.idx = [feature_names.index(c) for c in BASELINE_COLS]
        self.sk = None

    def fit(self, X, y, idx=None):
        Xs = X[:, self.idx]
        self.med = np.median(Xs, axis=0)
        self.mad = np.median(np.abs(Xs - self.med), axis=0) * 1.4826 + 1e-9
        return self

    def score(self, X, idx=None):
        return np.abs((X[:, self.idx] - self.med) / self.mad).max(axis=1)

    def artifacts(self):
        return {"robust_z_stats.json": {"median": self.med.tolist(),
                                        "mad": self.mad.tolist()}}


class IForest:
    """Semi-supervised: trained on normal rows only (label == 0)."""
    name = "isolation_forest"
    params = {"n_estimators": 200, "contamination": "auto"}

    def __init__(self, feature_names):
        self.sk = None

    def fit(self, X, y, idx=None):
        self.sk = IsolationForest(n_estimators=200, random_state=SEED,
                                  n_jobs=-1).fit(X[y == 0])
        return self

    def score(self, X, idx=None):
        return -self.sk.score_samples(X)     # higher = more anomalous

    def artifacts(self):
        return {}


class HISTGBM:
    """Supervised gradient boosting: P(RLF within horizon)."""
    name = "hist_gradient_boosting"
    params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.sk = None

    def fit(self, X, y, idx=None):
        pos = max(int((y == 1).sum()), 1)
        neg = int((y == 0).sum())
        self.sk = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=SEED)
        self.sk.fit(X, y)
        return self

    def score(self, X, idx=None):
        return self.sk.predict_proba(X)[:, 1]

    def artifacts(self):
        if hasattr(self.sk, "feature_importances_"):
            imp = sorted(zip(self.feature_names, self.sk.feature_importances_),
                         key=lambda kv: -kv[1])[:15]
            return {"feature_importance_top15.json":
                    {k: float(v) for k, v in imp}}
        return {}

class XGBM:
    """Supervised gradient boosting: P(RLF within horizon)."""
    name = "xgboost"
    params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.sk = None

    def fit(self, X, y, idx=None):
        pos = max(int((y == 1).sum()), 1)
        neg = int((y == 0).sum())
        
        self.sk = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(np.sqrt(neg / pos)),
            eval_metric="aucpr", n_jobs=-1, random_state=SEED)
        self.sk.fit(X, y)
        return self

    def score(self, X, idx=None):
        return self.sk.predict_proba(X)[:, 1]

    def artifacts(self):
        if hasattr(self.sk, "feature_importances_"):
            imp = sorted(zip(self.feature_names, self.sk.feature_importances_),
                         key=lambda kv: -kv[1])[:15]
            return {"feature_importance_top15.json":
                    {k: float(v) for k, v in imp}}
        return {}


class _GRUNet(nn.Module):
    """GRU over a (batch, window, channels) tensor -> one logit per sample."""
 
    def __init__(self, n_in, hidden):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
 
    def forward(self, x):
        _, h = self.gru(x)               # h: (1, batch, hidden) = last state
        return self.head(h[-1]).squeeze(-1)
 
 
class GRUModel:
    """Supervised GRU: reads the last `window` seconds of raw KPIs of one UE
    and outputs P(RLF within horizon). Same label and loss idea as XGBoost
    (binary cross-entropy with a sqrt(neg/pos) positive-class weight).
 
    Windows are built from the FULL dataframe (backward-looking only, same
    history the engineered features use), so cell hold-out folds are not
    fragmented. Inputs are standardised with statistics of the training rows only.
    The last 15 % of the training time range is used as validation to pick the
    best epoch (by PR-AUC).
    """
    name = "gru_supervised"
    window, hidden, epochs, batch, lr = 30, 32, 12, 512, 2e-3
    params = {}
    _seq = _start = _t = None            # filled by prepare()
 
    @classmethod
    def configure(cls, window, epochs):
        cls.window, cls.epochs = window, epochs
        cls.params = {"window_s": window, "hidden": cls.hidden,
                      "epochs": epochs, "batch": cls.batch, "lr": cls.lr,
                      "seq_channels": len(SEQ_COLS)}
 
    @classmethod
    def prepare(cls, df):
        """Cache the raw sequence matrix and where each UE's contiguous block starts."""
        cls._seq = np.nan_to_num(df[SEQ_COLS].to_numpy(np.float32))
        ue, t = df["ue_id"].to_numpy(), df["t"].to_numpy()
        new = np.ones(len(df), dtype=bool)
        new[1:] = (ue[1:] != ue[:-1]) | (np.diff(t) != 1)   # new UE or time gap
        pos = np.arange(len(df))
        cls._start = np.maximum.accumulate(np.where(new, pos, 0))
        cls._t = t
 
    def __init__(self, feature_names):
        self.sk = None                   # not a scikit-learn model
        self.net = None
 
    # -- helpers ---------------------------------------------------------
    def _windows(self, rows):
        """(len(rows), window, channels) standardised tensor; early seconds of a
        block are padded by repeating the block's first second."""
        off = np.arange(self.window - 1, -1, -1)
        widx = np.maximum(rows[:, None] - off[None, :], self._start[rows][:, None])
        x = (self._seq[widx] - self.mu) / self.sd
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
 
    def _predict(self, rows):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                logits = self.net(self._windows(rows[i:i + 4096]))
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out)
 
    # -- interface -------------------------------------------------------
    def fit(self, X, y, idx=None):
        if idx is None:
            raise ValueError("GRUModel needs `idx` (row positions in the dataframe)")
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        rows = np.asarray(idx)
        y = np.asarray(y)
 
        seq_tr = self._seq[rows]
        self.mu, self.sd = seq_tr.mean(0), seq_tr.std(0) + 1e-6
 
        # Time-based validation split inside the training rows.
        t = self._t[rows]
        cut = t.min() + 0.85 * (t.max() - t.min())
        is_tr = t < cut
        tr_rows, tr_y = rows[is_tr], y[is_tr]
        va_rows, va_y = rows[~is_tr], y[~is_tr]
        use_val = tr_y.sum() > 0 and va_y.sum() > 0
        if not use_val:                  # too few positives: train on everything
            tr_rows, tr_y = rows, y
 
        pos = max(int(tr_y.sum()), 1)
        neg = int(len(tr_y) - tr_y.sum())
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(np.sqrt(neg / pos))))
        self.net = _GRUNet(len(SEQ_COLS), self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
 
        best_ap, best_state = -1.0, None
        for _ in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(tr_rows))
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                xb = self._windows(tr_rows[b])
                yb = torch.from_numpy(tr_y[b].astype(np.float32))
                opt.zero_grad()
                lossf(self.net(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if use_val:                  # keep the best epoch
                ap = average_precision_score(va_y, self._predict(va_rows))
                if ap > best_ap:
                    best_ap, best_state = ap, copy.deepcopy(self.net.state_dict())
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self
 
    def score(self, X, idx=None):
        return self._predict(np.asarray(idx))
 
    def artifacts(self):
        return {"gru_scaler.json": {"seq_cols": SEQ_COLS, "window_s": self.window,
                                    "mean": self.mu.tolist(),
                                    "std": self.sd.tolist()}}
 
    def log_model(self, name="model"):
        """Log the network to MLflow under `name`.
 
        Recent MLflow defaults to the 'pt2' export format, which needs an
        input_example and can fail on recurrent layers, so we ask for plain
        pickle where supported. If logging still fails, fall back to saving the
        raw weights (state_dict), which torch.load can always read back.
        """
        import os
        import tempfile
        import mlflow.pytorch
        p = inspect.signature(mlflow.pytorch.log_model).parameters
        kw = {"name" if "name" in p else "artifact_path": name}
        if "serialization_format" in p:
            kw["serialization_format"] = "pickle"
        try:
            mlflow.pytorch.log_model(self.net, **kw)
        except Exception as exc:
            print(f"[{self.name}] mlflow.pytorch.log_model failed ({exc}); "
                  "saving the state_dict instead")
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "gru_state_dict.pt")
                torch.save(self.net.state_dict(), path)
                mlflow.log_artifact(path)


FAMILIES = [RobustZ, IForest, HISTGBM, XGBM, GRUModel]


# ----------------------------------------------------------------- splits ---
def time_splits(df, horizon_s, n_splits=4, start_frac=0.5):
    """Rolling-origin splits with an embargo of `horizon_s` between train/test."""
    t = df["t"].to_numpy()
    tmin, tmax = t.min(), t.max()
    span = tmax - tmin
    step = (1 - start_frac) / n_splits
    for k in range(n_splits):
        train_end = tmin + span * (start_frac + k * step)
        test_start = train_end + horizon_s
        test_end = test_start + span * step
        yield np.flatnonzero(t < train_end), \
            np.flatnonzero((t >= test_start) & (t < test_end))


def cell_holdout_split(df, horizon_s, frac=0.25, seed=0):
    """Train on some cells (early period), test on unseen cells (later period)."""
    rng = np.random.default_rng(seed)
    cells = np.sort(df["cell_id"].unique())
    held = rng.choice(cells, max(1, int(len(cells) * frac)), replace=False)
    t = df["t"].to_numpy()
    cutoff = t.min() + (t.max() - t.min()) * 0.7
    is_held = df["cell_id"].isin(held).to_numpy()
    return (np.flatnonzero(~is_held & (t < cutoff)),
            np.flatnonzero(is_held & (t >= cutoff + horizon_s)))


# ---------------------------------------------------------------- metrics ---
def evaluate(te_df, y, s, horizon_s):
    """Operational metrics at the threshold giving TARGET_FPR on normal rows."""
    out = {"pr_auc": float(average_precision_score(y, s))}
    thr = np.quantile(s[y == 0], 1 - TARGET_FPR)
    alert = s > thr
    out["recall_at_1pct_fpr"] = float(alert[y == 1].mean())

    t = te_df["t"].to_numpy()
    ue = te_df["ue_id"].to_numpy()
    rlf = te_df["rlf"].to_numpy()
    alert_times = {u: t[(ue == u) & alert] for u in np.unique(ue)}

    leads, events = [], np.flatnonzero(rlf == 1)
    for i in events:
        a = alert_times[ue[i]]
        w = a[(a >= t[i] - horizon_s) & (a < t[i])]
        if w.size:
            leads.append(t[i] - w.min())
    out["event_recall"] = len(leads) / len(events) if len(events) else np.nan
    out["median_lead_s"] = float(np.median(leads)) if leads else np.nan

    # Alerts merged into episodes: consecutive alert seconds are ONE alert.
    episodes = sum(1 + int((np.diff(a) > horizon_s).sum())
                   for a in alert_times.values() if a.size)
    out["alert_episodes_per_ue_hour"] = episodes / (len(te_df) / 3600.0)
    return out


def log_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        if v is not None and np.isfinite(v):
            mlflow.log_metric(prefix + k, float(v))


def mean_metrics(folds):
    return {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}


def log_sklearn_model(model, name="model"):
    """Log a scikit-learn-API model to the current MLflow run.
 
    - `name=` replaces the deprecated `artifact_path=` in recent MLflow; we pick
      whichever this installed version supports, so no deprecation warning.
    - serialization_format="cloudpickle" avoids the skops "untrusted types"
      error that recent MLflow raises for tree models. Cloudpickle files can run
      code when loaded, so only load models you created yourself (or trust).
    """
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {"name" if "name" in params else "artifact_path": name}
    if "serialization_format" in params:
        kwargs["serialization_format"] = "cloudpickle"
    mlflow.sklearn.log_model(model, **kwargs)

    
# -------------------------------------------------------------------- run ---
def run_family(Family, df, feat_cols, horizon_s, dataset_tag, n_splits):
    X = df[feat_cols].to_numpy(np.float32)
    y = df["label"].to_numpy()
    if hasattr(Family, "prepare"):       # sequence models cache the full series
        Family.prepare(df)

    # Name shown in the MLflow UI for the logged model (e.g. rlf_xgboost_h30s),
    # instead of the generic "model".
    model_name = f"rlf_{Family.name}_h{horizon_s}s"

    with mlflow.start_run(run_name=f"{Family.name}-h{horizon_s}s"):
        mlflow.set_tags({"dataset": dataset_tag, "task": "rlf_prediction",
                         "split": "rolling_origin_time", "horizon_s": horizon_s,
                         "feature_version": FEATURE_VERSION})
        # n_features = engineered features; the GRU instead logs seq_channels.
        mlflow.log_params({**Family.params, "n_features": len(feat_cols),
                           "n_rows": len(df), "target_fpr": TARGET_FPR})
 
        folds = []
        for tr, te in time_splits(df, horizon_s, n_splits):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            model = Family(feat_cols).fit(X[tr], y[tr], idx=tr)
            t0 = time.perf_counter()
            s = model.score(X[te], idx=te)
            infer_us = (time.perf_counter() - t0) / len(te) * 1e6
            m = evaluate(df.iloc[te], y[te], s, horizon_s)
            m["infer_us_per_sample"] = infer_us
            folds.append(m)
        if not folds:
            print(f"[{Family.name}] no usable folds (no positives)")
            return None
        summary = mean_metrics(folds)
        log_metrics(summary)
 
        # Generalisation to unseen cells.
        tr, te = cell_holdout_split(df, horizon_s)
        if y[tr].sum() > 0 and y[te].sum() > 0:
            model = Family(feat_cols).fit(X[tr], y[tr], idx=tr)
            ch = evaluate(df.iloc[te], y[te], model.score(X[te], idx=te), horizon_s)
            log_metrics(ch, prefix="cell_holdout_")
            summary["cell_holdout_pr_auc"] = ch["pr_auc"]
 
        # Final model on all data, plus artifacts that keep it portable.
        final = Family(feat_cols).fit(X, y, idx=np.arange(len(df)))
        mlflow.log_dict({"feature_version": FEATURE_VERSION,
                         "feature_columns": feat_cols}, "features.json")
        for fname, obj in final.artifacts().items():
            mlflow.log_dict(obj, fname)
        try:                             # logging the model must not kill the run
            if hasattr(final, "log_model"):
                final.log_model(name=model_name)
            elif final.sk is not None:
                log_sklearn_model(final.sk, name=model_name)
        except Exception as exc:
            print(f"[{Family.name}] model logging skipped: {exc}")
  
        summary["model"] = Family.name
        return summary
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, help="prediction horizon, s")
    ap.add_argument("--n-ues", type=int, default=60)
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--csv", default=None, help="real data with the same columns")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    ap.add_argument("--experiment", default="oran-rlf-prediction")
    ap.add_argument("--gru-window", type=int, default=60, help="GRU window, s")
    ap.add_argument("--gru-epochs", type=int, default=12)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    if args.csv:
        raw, tag = pd.read_csv(args.csv), f"csv:{args.csv}"
    else:
        raw, tag = (generate_dataset(args.n_ues, args.seconds, seed=args.seed),
                    f"synthetic-seed{args.seed}")
    df = build_dataset(raw, args.horizon)
    feat_cols = feature_columns(df)
    print(f"rows={len(df):,} features={len(feat_cols)} "
          f"positive_rate={df['label'].mean():.3%} rlf_events={int(df['rlf'].sum())}")
   
    GRUModel.configure(args.gru_window, args.gru_epochs)
    
    results = [r for r in (run_family(F, df, feat_cols, args.horizon, tag,
                                      args.n_splits) for F in FAMILIES) if r]
    table = pd.DataFrame(results).set_index("model").round(3)
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
