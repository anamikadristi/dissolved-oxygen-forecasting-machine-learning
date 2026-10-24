"""Step 4: forecast daily minimum DO 1, 2 and 3 days ahead at each site.

Models (all fit on the training block only; validation block is used for
early stopping and small hyperparameter choices; the test block is touched
once at the end):
  persistence     DOmin(t+h) = DOmin(t)
  climatology     smoothed day-of-year mean of training DOmin
  linear          ordinary least squares on standardized features
  random_forest   scikit-learn RandomForestRegressor
  lightgbm        LightGBM gradient boosting (small grid, early stopping)
  lstm            PyTorch LSTM predictions read from step 3 (lstm_predictions.csv)

Tabular ML models predict the change from today's DOmin, which is then added back.

Writes outputs/tables/forecast_metrics.csv, outputs/tables/test_predictions.csv
and outputs/tables/model_settings.csv.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from do_config import SITES, HORIZONS, TAB, SEED
from do_common import (site_frame, reg_metrics, climatology_fit, climatology_predict,
                       Imputer, fit_lgbm)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows, preds, settings = [], [], []
    LST = pd.read_csv(TAB / "lstm_predictions.csv", parse_dates=["issue_date"])
    for s in SITES:
        t0 = time.time()
        sf = site_frame(s["short"])
        F, Y, X, ok, sp = sf["F"], sf["Y"], sf["X"], sf["ok"], sf["split"]
        print(f"\n== {s['short']}: features={F.shape[1]} split val={sf['d1'].date()} "
              f"test={sf['d2'].date()}")

        L = LST[LST.site == s["short"]].set_index(["block", "horizon", "issue_date"]).sort_index()["lstm"]
        for h in HORIZONS:
            use = ok & Y[h].notna()
            tr, va, te = (use & (sp == "train")), (use & (sp == "val")), (use & (sp == "test"))
            base = F["do_min"]
            ytr, yva = (Y.loc[tr, h] - base[tr]).values, (Y.loc[va, h] - base[va]).values
            imp = Imputer().fit(F[tr])
            Ztr, Zva, Zte = imp.transform(F[tr]), imp.transform(F[va]), imp.transform(F[te])
            P = {}
            for blk, mask in [("val", va), ("test", te)]:
                P[(blk, "persistence")] = base[mask].values
            clim = climatology_fit(F.index[tr] + pd.Timedelta(days=h), Y.loc[tr, h].values)
            for blk, mask in [("val", va), ("test", te)]:
                P[(blk, "climatology")] = climatology_predict(
                    clim, F.index[mask] + pd.Timedelta(days=h))
            lr = LinearRegression().fit(Ztr, ytr)
            P[("val", "linear")] = base[va].values + lr.predict(Zva)
            P[("test", "linear")] = base[te].values + lr.predict(Zte)
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                       n_jobs=-1, random_state=SEED).fit(Ztr, ytr)
            P[("val", "random_forest")] = base[va].values + rf.predict(Zva)
            P[("test", "random_forest")] = base[te].values + rf.predict(Zte)
            gb, gbset = fit_lgbm(F[tr], ytr, F[va], yva)
            P[("val", "lightgbm")] = base[va].values + gb.predict(F[va])
            P[("test", "lightgbm")] = base[te].values + gb.predict(F[te])
            settings.append(dict(site=s["short"], model="lightgbm", horizon=h,
                                 setting=str(gbset)))
            for blk, mask in [("val", va), ("test", te)]:
                P[(blk, "lstm")] = L.loc[(blk, h)].reindex(F.index[mask]).values
            for blk, mask in [("val", va), ("test", te)]:
                obs = Y.loc[mask, h].values
                for mname in ["persistence", "climatology", "linear", "random_forest",
                              "lightgbm", "lstm"]:
                    r = reg_metrics(obs, P[(blk, mname)])
                    rows.append(dict(site=s["short"], horizon=h, block=blk, model=mname, **r,
                                     n_train=int(tr.sum())))
                if blk == "test":
                    df = pd.DataFrame({"site": s["short"], "issue_date": F.index[mask].date,
                                       "horizon": h, "obs": obs})
                    for mname in ["persistence", "climatology", "linear", "random_forest",
                                  "lightgbm", "lstm"]:
                        df[mname] = np.round(P[(blk, mname)], 3)
                    preds.append(df)
            t = pd.DataFrame([r for r in rows if r["site"] == s["short"] and r["horizon"] == h
                              and r["block"] == "test"])
            print(f" h={h} n_train={int(tr.sum())} n_test={int(te.sum())} | " +
                  " ".join(f"{m}:{v:.3f}" for m, v in zip(t.model, t.RMSE)))
        print(f" ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).round(4).to_csv(TAB / "forecast_metrics.csv", index=False)
    pd.concat(preds).to_csv(TAB / "test_predictions.csv", index=False)
    st = pd.concat([pd.DataFrame(settings), pd.read_csv(TAB / "lstm_settings.csv")])
    st.to_csv(TAB / "model_settings.csv", index=False)


if __name__ == "__main__":
    main()
