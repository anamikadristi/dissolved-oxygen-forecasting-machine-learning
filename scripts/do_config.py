"""Shared configuration: sites, parameters, paths and split settings."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG = REPO / "outputs" / "figures"
TAB = REPO / "outputs" / "tables"

END_DATE = "2026-10-22"

# Selected USGS continuous water-quality monitors (chosen from the NWIS site
# service for multi-year dissolved oxygen records, see README).
SITES = [
    {"site_no": "07380330", "short": "bayou_perot",
     "name": "Bayou Perot at Point Legard near Cutoff, LA",
     "setting": "Barataria Basin brackish estuary", "state": "LA",
     "source": "iv", "start": "2016-06-16", "weather": "USW00012916"},
    {"site_no": "07386880", "short": "vermilion",
     "name": "Vermilion River at Surrey St. at Lafayette, LA",
     "setting": "freshwater coastal-plain river", "state": "LA",
     "source": "iv", "start": "2018-03-23", "weather": "USW00013976"},
    {"site_no": "295826095082200", "short": "lake_houston",
     "name": "Lake Houston S Union Pacific RR Bridge near Houston, TX",
     "setting": "water-supply reservoir, Houston area", "state": "TX",
     "source": "dv", "start": "2008-09-30", "weather": "USW00012960"},
    {"site_no": "08068500", "short": "spring_creek",
     "name": "Spring Creek near Spring, TX",
     "setting": "urbanizing Houston-area stream", "state": "TX",
     "source": "dv", "start": "1999-12-10", "weather": "USW00012960"},
]
WEATHER_STATIONS = {s["short"]: s["weather"] for s in SITES}
WEATHER_NAMES = {"USW00012916": "New Orleans Intl Airport, LA",
                 "USW00013976": "Lafayette Regional Airport, LA",
                 "USW00012960": "Houston Intercontinental Airport, TX"}

# USGS parameter codes
PARAMS = {"00300": "do", "00010": "wtemp", "00095": "spc", "00400": "ph",
          "63680": "turb", "00060": "q", "00065": "gage", "00480": "sal"}
IV_PARAMS = list(PARAMS)
DV_PARAMS = list(PARAMS)

HORIZONS = [1, 2, 3]
HYPOXIA_THRESHOLDS = [2.0, 4.0]
# Time-ordered split fractions (by date, per site). A gap of max(HORIZONS)
# days is dropped at each boundary so no target overlaps the next split.
SPLIT = (0.70, 0.15, 0.15)
SEQ_LEN = 14  # days of history for the LSTM
SEED = 42
