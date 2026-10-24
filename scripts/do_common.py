"""Shared helpers: loading, feature engineering, splits, metrics, models."""
import itertools

import numpy as np
import pandas as pd

from do_config import DATA, HORIZONS, SPLIT, SEQ_LEN, SEED

# Optional site variables (used when the site measures them)
OPTIONAL = ["spc_mean", "sal_mean", "ph_mean", "ph_range", "turb_log", "q_log",
            "gage_mean", "gage_d1"]
WEATHER = ["air_tmax", "air_tmin", "precip", "precip_3d", "precip_7d", "wind", "wind_3d"]
# Features every site has (used for the cross-site transfer test)
COMMON = (["do_min", "do_mean", "do_max", "do_range", "do_d1", "do_roll3", "do_roll7"]
          + [f"do_min_l{k}" for k in range(1, 7)]
          + ["wtemp_mean", "wtemp_max", "wtemp_d1", "wtemp_roll7", "doy_sin", "doy_cos"]
          + WEATHER)
# Daily variables fed to the LSTM as a sequence
SEQ_BASE = ["do_min", "do_mean", "do_max", "wtemp_mean", "wtemp_max", "doy_sin",
            "doy_cos"] + ["air_tmax", "air_tmin", "precip", "wind"]


def load_site(short):
    d = pd.read_csv(DATA / f"daily_{short}.csv", index_col="date", parse_dates=True)
    return d.asfreq("D")


def daily_variables(d):
    """Derived daily variables known at the end of day t."""
    x = pd.DataFrame(index=d.index)
    x["do_min"], x["do_mean"], x["do_max"] = d["do_min"], d.get("do_mean"), d.get("do_max")
    if x["do_mean"].isna().all():
        x["do_mean"] = np.nan
    x["do_range"] = x["do_max"] - x["do_min"]
    x["wtemp_mean"] = d.get("wtemp_mean", d.get("wtemp_max"))
    x["wtemp_max"] = d.get("wtemp_max")
    for c in ["spc_mean", "sal_mean", "ph_mean", "gage_mean"]:
        if c in d:
            x[c] = d[c]
    if "ph_mean" not in d and "ph_max" in d:
        x["ph_mean"] = (d["ph_max"] + d["ph_min"]) / 2
    if "ph_max" in d:
        x["ph_range"] = d["ph_max"] - d["ph_min"]
    if "turb_mean" in d:
        x["turb_log"] = np.log1p(d["turb_mean"].clip(lower=0))
    if "q_mean" in d:
        x["q_log"] = np.log10(d["q_mean"].clip(lower=0.01))
    if "gage_mean" in d:
        x["gage_d1"] = d["gage_mean"].diff()
    for c in ["air_tmax", "air_tmin", "precip", "wind"]:
        x[c] = d[c]
    doy = d.index.dayofyear
    x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return x


def make_features(d):
    """Tabular features for issue day t and targets do_min(t+h)."""
    x = daily_variables(d)
    f = x.copy()
    for k in range(1, 7):
        f[f"do_min_l{k}"] = x["do_min"].shift(k)
    f["do_d1"] = x["do_min"] - x["do_min"].shift(1)
    f["do_roll3"] = x["do_min"].rolling(3, min_periods=2).mean()
    f["do_roll7"] = x["do_min"].rolling(7, min_periods=4).mean()
    f["wtemp_d1"] = x["wtemp_mean"].diff()
    f["wtemp_roll7"] = x["wtemp_mean"].rolling(7, min_periods=4).mean()
    f["precip_3d"] = x["precip"].rolling(3, min_periods=1).sum()
    f["precip_7d"] = x["precip"].rolling(7, min_periods=1).sum()
    f["wind_3d"] = x["wind"].rolling(3, min_periods=1).mean()
    feats = [c for c in COMMON if c in f] + [c for c in OPTIONAL if c in f]
    feats = [c for c in feats if f[c].notna().mean() > 0.3]
    y = pd.DataFrame({h: x["do_min"].shift(-h) for h in HORIZONS}, index=x.index)
    return f[feats], y, x


def split_dates(index_valid):
    """Time-ordered split points from the days that have a usable row."""
    idx = index_valid.sort_values()
    n = len(idx)
    d1 = idx[int(n * SPLIT[0])]
    d2 = idx[int(n * (SPLIT[0] + SPLIT[1]))]
    return d1, d2


def assign_split(index, d1, d2, gap=max(HORIZONS)):
    """train: t + gap < d1; val: d1 <= t and t + gap < d2; test: t >= d2.
    Rows whose target day would fall in the next block are dropped, so no
    validation or test target is ever seen during training."""
    s = pd.Series("drop", index=index)
    g = pd.Timedelta(days=gap)
    s[index + g < d1] = "train"
    s[(index >= d1) & (index + g < d2)] = "val"
    s[index >= d2] = "test"
    return s


def site_frame(short):
    d = load_site(short)
    F, Y, X = make_features(d)
    ok = F["do_min"].notna() & Y.notna().any(axis=1)
    d1, d2 = split_dates(F.index[ok])
    sp = assign_split(F.index, d1, d2)
    return dict(F=F, Y=Y, X=X, ok=ok, split=sp, d1=d1, d2=d2, raw=d)


# ---------------- metrics ----------------
def nse(o, p):
    o, p = np.asarray(o), np.asarray(p)
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(o, p):
    o, p = np.asarray(o), np.asarray(p)
    r = np.corrcoef(o, p)[0, 1]
    alpha = p.std() / o.std()
    beta = p.mean() / o.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def reg_metrics(o, p):
    o, p = np.asarray(o), np.asarray(p)
    return dict(n=len(o), RMSE=float(np.sqrt(np.mean((o - p) ** 2))),
                MAE=float(np.mean(np.abs(o - p))), NSE=float(nse(o, p)), KGE=float(kge(o, p)))


def climatology_fit(dates, values, window=15):
    """Smoothed day-of-year mean of daily minimum DO from training data only."""
    s = pd.Series(np.asarray(values), index=pd.DatetimeIndex(dates).dayofyear)
    m = s.groupby(level=0).mean().reindex(range(1, 367))
    ext = pd.concat([m.iloc[-window:], m, m.iloc[:window]])
    sm = ext.rolling(2 * window + 1, center=True, min_periods=5).mean().iloc[window:-window]
    sm.index = range(1, 367)
    return sm.interpolate(limit_direction="both")


def climatology_predict(clim, dates):
    return clim.reindex(pd.DatetimeIndex(dates).dayofyear).values


class Imputer:
    """Train-median imputation plus standardization (fit on training rows)."""

    def fit(self, X):
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        return self

    def transform(self, X):
        return ((X.fillna(self.med) - self.mu) / self.sd).values


def sequences(X, seq_vars, index_rows, imp_mu, imp_sd, seq_len=SEQ_LEN):
    """Build (n, seq_len, k) arrays ending at each issue day; missing -> 0
    after standardization, plus a missing-DO indicator channel."""
    Z = (X[seq_vars] - imp_mu) / imp_sd
    miss = X["do_min"].isna().astype(float).values[:, None]
    A = np.concatenate([Z.fillna(0).values, miss], axis=1).astype(np.float32)
    pos = X.index.get_indexer(index_rows)
    out = np.zeros((len(pos), seq_len, A.shape[1]), dtype=np.float32)
    for i, p in enumerate(pos):
        lo = p - seq_len + 1
        if lo < 0:
            out[i, -lo:] = A[: p + 1]
        else:
            out[i] = A[lo: p + 1]
    return out


def fit_lgbm(Xtr, ytr, Xva, yva):
    """LightGBM regressor: small grid over tree size, early stopping on the
    validation block; returns the model with the lowest validation RMSE.
    (Imported here so the PyTorch script never loads LightGBM.)"""
    import lightgbm as lgb
    best = None
    for nl, mcs in itertools.product([7, 15, 31], [20, 50]):
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=nl,
                              min_child_samples=mcs, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                              verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        rm = np.sqrt(np.mean((m.predict(Xva) - yva) ** 2))
        if best is None or rm < best[0]:
            best = (rm, m, dict(num_leaves=nl, min_child_samples=mcs,
                                n_trees=m.best_iteration_))
    return best[1], best[2]
