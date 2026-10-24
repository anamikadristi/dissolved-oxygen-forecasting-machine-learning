"""Step 2: build one clean daily table per site from the raw downloads.

- USGS daily values (TX sites): daily min / mean / max as published by USGS.
- USGS 15-minute values (LA sites): aggregated to local-calendar-day
  min / mean / max; a day is kept only if it has at least 75 percent of the
  usual number of readings for that parameter.
- Values flagged as equipment problems, ice, or similar are dropped; approved
  (A) and provisional (P) values are both kept and the share is reported.
- NOAA GHCN-Daily weather from the nearest major airport is joined by date.

Writes data/daily_<site>.csv and data/sites.csv.

Usage:
    python scripts/02_build_daily_dataset.py --raw-dir raw_data
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from do_config import SITES, PARAMS, DATA, WEATHER_NAMES

BAD_QUAL = {"Eqp", "Ice", "Mnt", "Dis", "Ssn", "Bkw", "Fld", "Pr", "Rat", "Zfl",
            "***", "Dry", "Tst"}


def parse_series(path):
    """Return list of (param, stat, method, DataFrame[dateTime, value, qual])."""
    txt = Path(path).read_text()
    if not txt.strip():
        return []
    js = json.loads(txt)
    out = []
    for ts in js["value"]["timeSeries"]:
        p = ts["variable"]["variableCode"][0]["value"]
        nod = ts["variable"].get("noDataValue", -999999.0)
        opt = ts["variable"]["options"]["option"]
        stat = next((o.get("optionCode") for o in opt if o.get("name") == "Statistic"), "iv")
        for v in ts["values"]:
            meth = v["method"][0].get("methodDescription", "") if v.get("method") else ""
            rows = v["value"]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.loc[df["value"] == nod, "value"] = np.nan
            df["qual"] = df["qualifiers"].apply(lambda q: ",".join(q))
            df = df[~df["qualifiers"].apply(lambda q: any(x in BAD_QUAL for x in q))]
            out.append((p, stat, meth, df[["dateTime", "value", "qual"]].dropna()))
    return out


def pick_longest(chunks):
    """Several method series can exist (e.g., multiple depths); keep the
    series key with the most records across all files."""
    by = {}
    for p, stat, meth, df in chunks:
        by.setdefault((p, stat, meth), []).append(df)
    best = {}
    for (p, stat, meth), dfs in by.items():
        n = sum(len(d) for d in dfs)
        if (p, stat) not in best or n > best[(p, stat)][0]:
            best[(p, stat)] = (n, meth, pd.concat(dfs))
    return best


def build_dv(site, raw):
    chunks = []
    for f in sorted(glob.glob(str(raw / "usgs" / f"dv_{site}_*.json"))):
        chunks += parse_series(f)
    best = pick_longest(chunks)
    cols, meta = {}, {}
    sname = {"00001": "max", "00002": "min", "00003": "mean"}
    prov = []
    for (p, stat), (n, meth, df) in best.items():
        if p not in PARAMS or stat not in sname:
            continue
        df["date"] = pd.to_datetime(df["dateTime"].str[:10])
        s = df.drop_duplicates("date").set_index("date")["value"]
        cols[f"{PARAMS[p]}_{sname[stat]}"] = s
        meta[f"{PARAMS[p]}_{sname[stat]}"] = meth or "default"
        if p == "00300":
            prov.append(df["qual"].str.contains("P").mean())
    out = pd.DataFrame(cols).sort_index()
    return out, meta, float(np.mean(prov)) if prov else np.nan


def build_iv(site, raw):
    chunks = []
    for f in sorted(glob.glob(str(raw / "usgs" / f"iv_{site}_*.json"))):
        chunks += parse_series(f)
    best = pick_longest(chunks)
    cols, meta, prov = {}, {}, np.nan
    for (p, stat), (n, meth, df) in best.items():
        if p not in PARAMS:
            continue
        # local clock date from the ISO string (offset kept by USGS)
        df["date"] = pd.to_datetime(df["dateTime"].str[:10])
        df = df.drop_duplicates("dateTime")
        g = df.groupby("date")["value"]
        cnt = g.size()
        need = 0.75 * cnt.median()
        ok = cnt[cnt >= need].index
        name = PARAMS[p]
        agg = g.agg(["min", "mean", "max"]).loc[ok]
        for k in ["min", "mean", "max"]:
            cols[f"{name}_{k}"] = agg[k]
        meta[name] = meth or "default"
        if p == "00300":
            prov = df["qual"].str.contains("P").mean()
    out = pd.DataFrame(cols).sort_index()
    return out, meta, float(prov)


def load_weather(raw, station):
    w = pd.read_csv(raw / "noaa" / f"ghcnd_{station}.csv", parse_dates=["DATE"])
    w = w.set_index("DATE")[["TMAX", "TMIN", "PRCP", "AWND"]]
    w.columns = ["air_tmax", "air_tmin", "precip", "wind"]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw_data")
    raw = Path(ap.parse_args().raw_dir)
    DATA.mkdir(exist_ok=True)
    info = []
    for s in SITES:
        if s["source"] == "dv":
            d, meta, prov = build_dv(s["site_no"], raw)
        else:
            d, meta, prov = build_iv(s["site_no"], raw)
        d = d[d.index >= s["start"]]
        full = pd.date_range(d.index.min(), d.index.max(), freq="D")
        d = d.reindex(full)
        d.index.name = "date"
        # drop physically implausible DO values
        for c in [c for c in d if c.startswith("do_")]:
            d.loc[(d[c] < 0) | (d[c] > 25), c] = np.nan
        w = load_weather(raw, s["weather"])
        d = d.join(w, how="left")
        # trim to first/last valid DO minimum
        v = d["do_min"].dropna()
        d = d.loc[v.index.min():v.index.max()]
        keep = [c for c in d.columns if d[c].notna().mean() > 0.2]
        d = d[keep].round(3)
        d.to_csv(DATA / f"daily_{s['short']}.csv")
        info.append({
            "short": s["short"], "site_no": s["site_no"], "name": s["name"],
            "state": s["state"], "setting": s["setting"], "usgs_source": s["source"],
            "first_day": d.index.min().date(), "last_day": d.index.max().date(),
            "days_in_span": len(d), "days_with_do_min": int(d["do_min"].notna().sum()),
            "do_min_median": round(d["do_min"].median(), 2),
            "pct_days_do_lt2": round(100 * (d["do_min"] < 2).sum() / d["do_min"].notna().sum(), 1),
            "pct_days_do_lt4": round(100 * (d["do_min"] < 4).sum() / d["do_min"].notna().sum(), 1),
            "share_provisional_do": round(prov, 3) if prov == prov else "",
            "variables": " ".join(sorted({c.rsplit('_', 1)[0] for c in keep})),
            "noaa_station": s["weather"], "noaa_name": WEATHER_NAMES[s["weather"]],
            "do_series_method": meta.get("do", meta.get("do_min", "")),
        })
        print(s["short"], len(d), "days;", info[-1]["days_with_do_min"], "with DO min;",
              "vars:", info[-1]["variables"], "| method:", info[-1]["do_series_method"])
    pd.DataFrame(info).to_csv(DATA / "sites.csv", index=False)
    print(pd.DataFrame(info).drop(columns=["name", "variables"]).to_string())


if __name__ == "__main__":
    main()
