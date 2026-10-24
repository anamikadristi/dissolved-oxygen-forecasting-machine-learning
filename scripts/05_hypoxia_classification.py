"""Step 5: classify next-day to 3-day-ahead hypoxia risk.

Label: daily minimum DO(t+h) below 2 mg/L or below 4 mg/L.
Same time-ordered train / validation / test blocks as the regression models.

Classifiers:
  persistence_rule   alarm if DOmin(t) is already below the threshold
  logistic           class_weight="balanced"
  random_forest      class_weight="balanced_subsample"
  lightgbm           scale_pos_weight = negatives / positives in training
  lstm_regression    alarm if the step 3 LSTM forecast is below the threshold

Class imbalance is handled with the weights above, and the probability cut
for the three probabilistic classifiers is chosen on the validation block to
maximize F1, then applied unchanged to the test block. A site and threshold
is only scored when the training block has at least 15 events and the test
block has at least 5 events.

Writes outputs/tables/hypoxia_metrics.csv and hypoxia_event_counts.csv.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score)

from do_config import SITES, HORIZONS, HYPOXIA_THRESHOLDS, TAB, SEED
from do_common import site_frame, Imputer

MIN_TRAIN_EVENTS, MIN_TEST_EVENTS = 15, 5


def best_cut(y, p):
    if y.sum() == 0:
        return 0.5
    cuts = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
    f = [f1_score(y, p >= c, zero_division=0) for c in cuts]
    return float(cuts[int(np.argmax(f))])


def score(y, prob, alarm):
    out = dict(precision=precision_score(y, alarm, zero_division=0),
               recall=recall_score(y, alarm, zero_division=0),
               F1=f1_score(y, alarm, zero_division=0))
    if prob is not None and 0 < y.sum() < len(y):
        out.update(ROC_AUC=roc_auc_score(y, prob), PR_AUC=average_precision_score(y, prob))
    return out


def main():
    LST = pd.read_csv(TAB / "lstm_predictions.csv", parse_dates=["issue_date"])
    rows, counts = [], []
    for s in SITES:
        sf = site_frame(s["short"])
        F, Y, ok, sp = sf["F"], sf["Y"], sf["ok"], sf["split"]
        L = LST[LST.site == s["short"]].set_index(["block", "horizon", "issue_date"]).sort_index()["lstm"]
        for thr in HYPOXIA_THRESHOLDS:
            for h in HORIZONS:
                use = ok & Y[h].notna()
                m = {k: use & (sp == k) for k in ["train", "val", "test"]}
                y = {k: (Y.loc[m[k], h] < thr).astype(int).values for k in m}
                c = dict(site=s["short"], threshold=thr, horizon=h,
                         **{f"events_{k}": int(y[k].sum()) for k in y},
                         **{f"n_{k}": int(len(y[k])) for k in y})
                counts.append(c)
                if y["train"].sum() < MIN_TRAIN_EVENTS or y["test"].sum() < MIN_TEST_EVENTS:
                    print(f"{s['short']} <{thr} h={h}: too few events {c}")
                    continue
                imp = Imputer().fit(F[m["train"]])
                Z = {k: imp.transform(F[m[k]]) for k in m}
                base = {k: F.loc[m[k], "do_min"].values for k in m}
                res = {}
                # persistence: score = -DOmin(t), alarm at the threshold itself
                res["persistence_rule"] = (-base["test"], (base["test"] < thr).astype(int))
                pos = y["train"].sum()
                spw = (len(y["train"]) - pos) / pos
                models = {
                    "logistic": LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0),
                    "random_forest": RandomForestClassifier(
                        n_estimators=400, min_samples_leaf=3, max_features=0.5,
                        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
                }
                for name, mdl in models.items():
                    mdl.fit(Z["train"], y["train"])
                    pv, pt = mdl.predict_proba(Z["val"])[:, 1], mdl.predict_proba(Z["test"])[:, 1]
                    res[name] = (pt, (pt >= best_cut(y["val"], pv)).astype(int))
                gb = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15,
                                        min_child_samples=20, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, scale_pos_weight=spw,
                                        random_state=SEED, verbose=-1)
                if y["val"].sum() > 0:
                    gb.fit(F[m["train"]], y["train"], eval_set=[(F[m["val"]], y["val"])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
                else:
                    gb.set_params(n_estimators=300).fit(F[m["train"]], y["train"])
                pv, pt = gb.predict_proba(F[m["val"]])[:, 1], gb.predict_proba(F[m["test"]])[:, 1]
                res["lightgbm"] = (pt, (pt >= best_cut(y["val"], pv)).astype(int))
                lp = L.loc[("test", h)].reindex(F.index[m["test"]]).values
                res["lstm_regression"] = (-lp, (lp < thr).astype(int))
                for name, (prob, alarm) in res.items():
                    rows.append(dict(site=s["short"], threshold=thr, horizon=h, model=name,
                                     events_test=int(y["test"].sum()), n_test=len(y["test"]),
                                     **score(y["test"], prob, alarm)))
                t = pd.DataFrame([r for r in rows if r["site"] == s["short"]
                                  and r["threshold"] == thr and r["horizon"] == h])
                print(f"{s['short']} <{thr} h={h} events test={y['test'].sum()}/{len(y['test'])} | "
                      + " ".join(f"{a}: AUC {b:.3f} F1 {c:.2f}" for a, b, c in
                                 zip(t.model, t.ROC_AUC, t.F1)))
    pd.DataFrame(rows).round(4).to_csv(TAB / "hypoxia_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(TAB / "hypoxia_event_counts.csv", index=False)


if __name__ == "__main__":
    main()
