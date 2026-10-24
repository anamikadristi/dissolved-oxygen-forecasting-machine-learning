"""Shared matplotlib style and colors (categorical order fixed, never cycled)."""
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

MODELS = ["persistence", "climatology", "linear", "random_forest", "lightgbm", "lstm"]
MODEL_LABEL = {"persistence": "Persistence", "climatology": "Climatology",
               "linear": "Linear regression", "random_forest": "Random forest",
               "lightgbm": "LightGBM", "lstm": "LSTM"}
MODEL_COLOR = {"persistence": "#9b9a95", "climatology": "#cfcdc6", "linear": AQUA,
               "random_forest": ORANGE, "lightgbm": BLUE, "lstm": VIOLET}
SITE_LABEL = {"bayou_perot": "Bayou Perot, LA", "vermilion": "Vermilion River, LA",
              "lake_houston": "Lake Houston, TX", "spring_creek": "Spring Creek, TX"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
        "lines.linewidth": 1.6,
    })
