"""Step 7: cross-site transfer test.

Question: does a model trained at one monitor forecast DO at a different
monitor it has never seen?

Uses LightGBM with only the features every site has (DO history, water
temperature, airport weather, season). For each horizon:
  - single-site transfer: train on site A's training block (early stopping on
    A's validation block), predict site B's test block (all A, B pairs;
    the diagonal A = B is the local model with the same reduced features);
  - leave-one-site-out pooled: train on the other three sites' training
    blocks (early stopping on their validation blocks), predict the held-out
    site's test block.
Persistence on the same test rows is reported for reference.

Writes outputs/tables/transfer_metrics.csv.
"""
import numpy as np
import pandas as pd

from do_config import SITES, HORIZONS, TAB
from do_common import site_frame, reg_metrics, fit_lgbm, COMMON


def blocks(sf, h, feats):
    F, Y, ok, sp = sf["F"], sf["Y"], sf["ok"], sf["split"]
    use = ok & Y[h].notna()
    out = {}
    for k in ["train", "val", "test"]:
        m = use & (sp == k)
        out[k] = (F.loc[m, feats], (Y.loc[m, h] - F.loc[m, "do_min"]).values,
                  F.loc[m, "do_min"].values, Y.loc[m, h].values)
    return out


def main():
    sfs = {s["short"]: site_frame(s["short"]) for s in SITES}
    feats = [c for c in COMMON if all(c in sf["F"] for sf in sfs.values())]
    print("common features:", feats)
    rows = []
    for h in HORIZONS:
        B = {k: blocks(sf, h, feats) for k, sf in sfs.items()}
        models = {}
        for a in B:
            models[a], _ = fit_lgbm(B[a]["train"][0], B[a]["train"][1],
                                    B[a]["val"][0], B[a]["val"][1])
        for a in B:
            for b in B:
                X, _, base, obs = B[b]["test"]
                r = reg_metrics(obs, base + models[a].predict(X))
                rows.append(dict(horizon=h, train_site=a, test_site=b, scheme="single_site", **r))
        for b in B:
            others = [a for a in B if a != b]
            Xtr = pd.concat([B[a]["train"][0] for a in others])
            ytr = np.concatenate([B[a]["train"][1] for a in others])
            Xva = pd.concat([B[a]["val"][0] for a in others])
            yva = np.concatenate([B[a]["val"][1] for a in others])
            m, _ = fit_lgbm(Xtr, ytr, Xva, yva)
            X, _, base, obs = B[b]["test"]
            rows.append(dict(horizon=h, train_site="other_three", test_site=b,
                             scheme="leave_one_site_out", **reg_metrics(obs, base + m.predict(X))))
            rows.append(dict(horizon=h, train_site="none", test_site=b, scheme="persistence",
                             **reg_metrics(obs, base)))
        t = pd.DataFrame([r for r in rows if r["horizon"] == h and r["scheme"] == "single_site"])
        print(f"h={h} RMSE matrix (rows=train, cols=test)")
        print(t.pivot(index="train_site", columns="test_site", values="RMSE").round(3))
    pd.DataFrame(rows).round(4).to_csv(TAB / "transfer_metrics.csv", index=False)
    print(pd.DataFrame(rows).query("scheme != 'single_site'").round(3).to_string())


if __name__ == "__main__":
    main()
