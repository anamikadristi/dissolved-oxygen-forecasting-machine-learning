"""Step 6: SHAP interpretation of the LightGBM 1-day-ahead models.

The LightGBM model for each site is refit exactly as in step 4 (same data,
same grid, same seed) and explained with shap.TreeExplainer on the test
block. The model target is the change in DOmin from today to tomorrow, so
SHAP values show what pushes tomorrow's minimum up or down relative to today.

Writes outputs/tables/shap_mean_abs.csv and two figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from do_config import SITES, TAB, FIG
from do_common import site_frame, fit_lgbm
from plot_style import apply_style, SITE_LABEL, INK, INK2, BLUE

H = 1


def main():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    rows, store = [], {}
    for s in SITES:
        sf = site_frame(s["short"])
        F, Y, ok, sp = sf["F"], sf["Y"], sf["ok"], sf["split"]
        use = ok & Y[H].notna()
        tr, va, te = use & (sp == "train"), use & (sp == "val"), use & (sp == "test")
        base = F["do_min"]
        gb, _ = fit_lgbm(F[tr], (Y.loc[tr, H] - base[tr]).values,
                         F[va], (Y.loc[va, H] - base[va]).values)
        sv = shap.TreeExplainer(gb).shap_values(F[te])
        m = np.abs(sv).mean(0)
        for f, v in zip(F.columns, m):
            rows.append(dict(site=s["short"], feature=f, mean_abs_shap=v))
        store[s["short"]] = (sv, F[te])
        top = pd.Series(m, index=F.columns).sort_values(ascending=False).head(5)
        print(s["short"], top.round(3).to_dict())
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby("site")["mean_abs_shap"].rank(ascending=False).astype(int)
    t.sort_values(["site", "rank"]).round(4).to_csv(TAB / "shap_mean_abs.csv", index=False)

    # Figure: top 10 features per site (mean |SHAP|)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    for ax, s in zip(axes, SITES):
        d = t[t.site == s["short"]].nsmallest(10, "rank").iloc[::-1]
        ax.barh(d.feature, d.mean_abs_shap, color=BLUE, height=0.6)
        ax.set_title(SITE_LABEL[s["short"]], fontsize=10, color=INK)
        ax.set_xlabel("mean |SHAP| (mg/L)", color=INK2)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("What drives the 1-day-ahead LightGBM forecast of the change in daily "
                 "minimum DO (test block)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_shap_importance.png", dpi=150)
    plt.close(fig)

    # Figure: beeswarm for the most hypoxia-prone site
    sv, Xte = store["vermilion"]
    plt.figure()
    shap.summary_plot(sv, Xte, max_display=12, show=False, plot_size=(7, 5))
    plt.title("Vermilion River: SHAP values, 1-day-ahead change in DOmin", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "fig06_shap_beeswarm_vermilion.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
