"""Step 8: summary tables and figures from the saved outputs of steps 2 to 7."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from do_config import SITES, HORIZONS, TAB, FIG
from do_common import site_frame
from plot_style import (apply_style, MODELS, MODEL_LABEL, MODEL_COLOR, SITE_LABEL, INK, INK2,
                        MUTED, BLUE, ORANGE, RED, YELLOW)

SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def tables():
    m = pd.read_csv(TAB / "forecast_metrics.csv")
    te = m[m.block == "test"]
    out = []
    for (site, h), g in te.groupby(["site", "horizon"], sort=False):
        g = g.set_index("model")
        r = dict(site=site, horizon=h, n_test=int(g.n.iloc[0]))
        for mod in MODELS:
            r[f"RMSE_{mod}"] = g.loc[mod, "RMSE"]
        for mod in MODELS:
            r[f"NSE_{mod}"] = g.loc[mod, "NSE"]
        for mod in MODELS:
            r[f"KGE_{mod}"] = g.loc[mod, "KGE"]
        ml = g.loc[["linear", "random_forest", "lightgbm", "lstm"], "RMSE"]
        r["best_model"] = g["RMSE"].idxmin()
        r["lstm_minus_lightgbm_RMSE"] = g.loc["lstm", "RMSE"] - g.loc["lightgbm", "RMSE"]
        r["lightgbm_skill_vs_persistence_pct"] = 100 * (1 - g.loc["lightgbm", "RMSE"] /
                                                        g.loc["persistence", "RMSE"])
        r["lstm_skill_vs_persistence_pct"] = 100 * (1 - g.loc["lstm", "RMSE"] /
                                                    g.loc["persistence", "RMSE"])
        out.append(r)
    comp = pd.DataFrame(out).round(3)
    comp.to_csv(TAB / "model_comparison_test.csv", index=False)

    # error by season (test block)
    p = pd.read_csv(TAB / "test_predictions.csv", parse_dates=["issue_date"])
    p["target_date"] = p.issue_date + pd.to_timedelta(p.horizon, unit="D")
    p["season"] = p.target_date.dt.month.map(SEASON)
    rows = []
    for (site, h, sea), g in p.groupby(["site", "horizon", "season"]):
        for mod in MODELS:
            e = g[mod] - g.obs
            rows.append(dict(site=site, horizon=h, season=sea, model=mod, n=len(g),
                             RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                             bias=np.mean(e)))
    sea = pd.DataFrame(rows).round(3)
    sea.to_csv(TAB / "error_by_season_test.csv", index=False)
    return comp, sea, p


def fig_timeseries():
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5))
    for ax, s in zip(axes, SITES):
        sf = site_frame(s["short"])
        x = sf["X"]["do_min"]
        ax.axvspan(sf["d1"], sf["d2"], color="#eeedea", lw=0)
        ax.axvspan(sf["d2"], x.index.max(), color="#dcdad4", lw=0)
        ax.plot(x.index, x.values, color=BLUE, lw=0.7)
        for thr, c in [(2, RED), (4, YELLOW)]:
            ax.axhline(thr, color=c, lw=1, ls="--")
        ax.set_ylabel("daily min DO (mg/L)")
        ax.set_title(f"{SITE_LABEL[s['short']]}  (USGS {s['site_no']})", loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.text(sf["d1"], ax.get_ylim()[1] * 0.92, " validation", color=INK2, fontsize=8)
        ax.text(sf["d2"], ax.get_ylim()[1] * 0.92, " test", color=INK2, fontsize=8)
    axes[0].text(0.995, 1.04, "dashed lines: 2 and 4 mg/L hypoxia thresholds",
                 transform=axes[0].transAxes, ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_daily_min_do_and_splits.png", dpi=150)
    plt.close(fig)


def fig_comparison(comp):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=False)
    w = 0.13
    for ax, s in zip(axes, SITES):
        g = comp[comp.site == s["short"]]
        for i, mod in enumerate(MODELS):
            ax.bar(np.arange(3) + (i - 2.5) * w, g[f"RMSE_{mod}"], width=w * 0.85,
                   color=MODEL_COLOR[mod], label=MODEL_LABEL[mod])
        ax.set_xticks(range(3), [f"{h} day" for h in HORIZONS])
        ax.set_title(SITE_LABEL[s["short"]], fontsize=10)
        ax.set_ylabel("test RMSE (mg/L)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=6, loc="upper left", bbox_to_anchor=(0, 1.25), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_model_comparison_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_season(sea):
    order = ["DJF", "MAM", "JJA", "SON"]
    show = ["persistence", "linear", "random_forest", "lightgbm", "lstm"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    w = 0.16
    for ax, s in zip(axes, SITES):
        g = sea[(sea.site == s["short"]) & (sea.horizon == 1)]
        for i, mod in enumerate(show):
            v = g[g.model == mod].set_index("season").reindex(order)["RMSE"]
            ax.bar(np.arange(4) + (i - 2) * w, v, width=w * 0.85, color=MODEL_COLOR[mod],
                   label=MODEL_LABEL[mod])
        n = g[g.model == "lightgbm"].set_index("season").reindex(order)["n"]
        ax.set_xticks(range(4), [f"{o}\nn={int(k) if k == k else 0}" for o, k in zip(order, n)])
        ax.set_title(SITE_LABEL[s["short"]], fontsize=10)
        ax.set_ylabel("1-day test RMSE (mg/L)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.25), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_error_by_season.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_example(p):
    g = p[(p.site == "vermilion") & (p.horizon == 1)].copy()
    g = g[(g.target_date >= "2026-04-01") & (g.target_date <= "2026-10-22")]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(g.target_date, g.obs, color=INK, lw=1.4, label="Observed")
    ax.plot(g.target_date, g.persistence, color=MODEL_COLOR["persistence"], lw=1,
            label="Persistence")
    ax.plot(g.target_date, g.lightgbm, color=BLUE, lw=1.2, label="LightGBM")
    ax.plot(g.target_date, g.lstm, color=MODEL_COLOR["lstm"], lw=1.2, ls="--", label="LSTM")
    ax.axhline(2, color=RED, lw=1, ls=":")
    ax.text(g.target_date.min(), 2.1, "2 mg/L", color=INK2, fontsize=8)
    ax.set_ylabel("daily min DO (mg/L)")
    ax.set_title("Vermilion River at Lafayette, LA: 1-day-ahead forecasts in the test block "
                 "(Apr to Oct 2026, provisional USGS data)", loc="left", fontsize=10)
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_example_forecast_vermilion.png", dpi=150)
    plt.close(fig)


def fig_hypoxia():
    t = pd.read_csv(TAB / "hypoxia_metrics.csv")
    t = t[t.horizon == 1]
    combos = [(r.site, r.threshold) for r in t[["site", "threshold"]].drop_duplicates().itertuples()]
    mods = ["persistence_rule", "logistic", "random_forest", "lightgbm", "lstm_regression"]
    lab = {"persistence_rule": "Persistence rule", "logistic": "Logistic (balanced)",
           "random_forest": "Random forest (balanced)", "lightgbm": "LightGBM (weighted)",
           "lstm_regression": "LSTM forecast < threshold"}
    col = {"persistence_rule": MODEL_COLOR["persistence"], "logistic": "#1baf7a",
           "random_forest": ORANGE, "lightgbm": BLUE, "lstm_regression": MODEL_COLOR["lstm"]}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    w = 0.16
    for ax, met in zip(axes, ["ROC_AUC", "F1"]):
        for i, mod in enumerate(mods):
            v = [t[(t.site == a) & (t.threshold == b) & (t.model == mod)][met].iloc[0]
                 for a, b in combos]
            ax.bar(np.arange(len(combos)) + (i - 2) * w, v, width=w * 0.85, color=col[mod],
                   label=lab[mod])
        ev = [int(t[(t.site == a) & (t.threshold == b)].events_test.iloc[0]) for a, b in combos]
        ax.set_xticks(range(len(combos)),
                      [f"{SITE_LABEL[a]}\nDO < {b:g} mg/L\n{e} test events" for (a, b), e in
                       zip(combos, ev)], fontsize=8)
        ax.set_ylabel(met.replace("_", "-") + " (test)")
        ax.set_ylim(0 if met == "F1" else 0.5, 1.0)
        ax.grid(axis="x", visible=False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=5, loc="upper center", fontsize=8)
    fig.suptitle("Hypoxia classification, 1 day ahead (test block)", y=0.955, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig07_hypoxia_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_transfer():
    t = pd.read_csv(TAB / "transfer_metrics.csv")
    t = t[t.horizon == 1]
    pers = t[t.scheme == "persistence"].set_index("test_site")["RMSE"]
    names = [s["short"] for s in SITES]
    single = t[t.scheme == "single_site"].pivot(index="train_site", columns="test_site",
                                                 values="RMSE").loc[names, names]
    loso = t[t.scheme == "leave_one_site_out"].set_index("test_site")["RMSE"].reindex(names)
    M = pd.concat([single, loso.to_frame().T.rename(index={"RMSE": "other_three"})])
    skill = 100 * (1 - M / pers.reindex(names).values)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lim = np.nanmax(np.abs(skill.values))
    im = ax.imshow(skill.values, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(skill.shape[0]):
        for j in range(skill.shape[1]):
            ax.text(j, i, f"{skill.values[i, j]:+.0f}%\n({M.values[i, j]:.2f})", ha="center",
                    va="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), [SITE_LABEL[n] for n in names], rotation=20, ha="right")
    ax.set_yticks(range(5), [SITE_LABEL[n] for n in names] + ["Other three (pooled)"])
    ax.set_xlabel("tested on (test block)")
    ax.set_ylabel("trained on")
    ax.grid(False)
    ax.axhline(3.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("RMSE skill vs persistence (%)")
    ax.set_title("Cross-site transfer, 1 day ahead (LightGBM, shared features)\n"
                 "cell: skill vs persistence, with RMSE in mg/L", fontsize=10, loc="center")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_cross_site_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    comp, sea, p = tables()
    fig_timeseries()
    fig_comparison(comp)
    fig_season(sea)
    fig_example(p)
    fig_hypoxia()
    fig_transfer()
    cols = ["site", "horizon"] + [f"RMSE_{m}" for m in MODELS] + ["best_model",
                                                                  "lstm_minus_lightgbm_RMSE"]
    print(comp[cols].to_string())


if __name__ == "__main__":
    main()
