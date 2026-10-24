"""Step 1: download public USGS water-quality and NOAA weather data.

All sources are keyless public web services:
  - USGS NWIS Site Service      https://waterservices.usgs.gov/nwis/site/
  - USGS NWIS Daily Values      https://waterservices.usgs.gov/nwis/dv/
  - USGS NWIS Instantaneous     https://nwis.waterservices.usgs.gov/nwis/iv/
  - NOAA NCEI Access Data (GHCN-Daily) https://www.ncei.noaa.gov/access/services/data/v1

Raw files go to --raw-dir (default: raw_data/, which is git-ignored).

Usage:
    python scripts/01_download_data.py --raw-dir raw_data
"""
import argparse
import json
import time
from pathlib import Path

import requests

from do_config import SITES, WEATHER_STATIONS, END_DATE, IV_PARAMS, DV_PARAMS

UA = {"User-Agent": "public-data-research-script (python requests)"}


def get(url, params, out, retries=4):
    if out.exists() and out.stat().st_size > 0:
        print("  exists:", out.name)
        return
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=300)
            if r.status_code == 404:
                print("  no data (404):", out.name)
                out.write_text("")
                return
            r.raise_for_status()
            out.write_bytes(r.content)
            print(f"  saved {out.name} ({len(r.content)/1e6:.1f} MB)")
            return
        except Exception as e:  # network hiccup, retry
            print("  retry", k + 1, e)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed: {url} {params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw_data")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    (raw / "usgs").mkdir(parents=True, exist_ok=True)
    (raw / "noaa").mkdir(parents=True, exist_ok=True)

    # Site metadata (series inventory) for each selected site
    for s in SITES:
        get("https://waterservices.usgs.gov/nwis/site/",
            {"format": "rdb", "sites": s["site_no"], "seriesCatalogOutput": "true",
             "siteStatus": "all"},
            raw / "usgs" / f"site_{s['site_no']}.rdb")

    for s in SITES:
        sid = s["site_no"]
        y0, y1 = int(s["start"][:4]), int(END_DATE[:4])
        if s["source"] == "dv":
            # Daily values (min, max, mean) computed and reviewed by USGS
            for y in range(y0, y1 + 1, 4):
                st = max(s["start"], f"{y}-01-01")
                en = min(END_DATE, f"{y+3}-12-31")
                get("https://waterservices.usgs.gov/nwis/dv/",
                    {"format": "json", "sites": sid, "parameterCd": ",".join(DV_PARAMS),
                     "statCd": "00001,00002,00003", "startDT": st, "endDT": en,
                     "siteStatus": "all"},
                    raw / "usgs" / f"dv_{sid}_{y}.json")
        else:
            # 15-minute instantaneous values, aggregated to daily in step 2
            for y in range(y0, y1 + 1):
                st = max(s["start"], f"{y}-01-01")
                en = min(END_DATE, f"{y}-12-31")
                get("https://nwis.waterservices.usgs.gov/nwis/iv/",
                    {"format": "json", "sites": sid, "parameterCd": ",".join(IV_PARAMS),
                     "startDT": st, "endDT": en, "siteStatus": "all"},
                    raw / "usgs" / f"iv_{sid}_{y}.json")

    # NOAA GHCN-Daily airport weather (keyless NCEI access service)
    start = min(s["start"] for s in SITES)
    for st_id in sorted(set(WEATHER_STATIONS.values())):
        get("https://www.ncei.noaa.gov/access/services/data/v1",
            {"dataset": "daily-summaries", "stations": st_id, "startDate": start,
             "endDate": END_DATE, "dataTypes": "TMAX,TMIN,PRCP,AWND",
             "format": "csv", "units": "metric", "includeStationName": "true"},
            raw / "noaa" / f"ghcnd_{st_id}.csv")

    (raw / "download_log.json").write_text(json.dumps(
        {"accessed": time.strftime("%Y-%m-%d"), "sites": SITES,
         "weather": WEATHER_STATIONS}, indent=2))
    print("done")


if __name__ == "__main__":
    main()
