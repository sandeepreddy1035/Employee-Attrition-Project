"""
Utility functions and project-wide configurations for Employee Attrition Prediction.
"""

import os
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Define base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "trained_models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIZ_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Ensure all directories exist
for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    VIZ_DIR,
    REPORTS_DIR,
    METRICS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    return logging.getLogger(name)


def set_plotting_theme():
    """Applies a consistent, clean, high-DPI style to Matplotlib and Seaborn plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.sans-serif"] = "DejaVu Sans", "Arial", "Helvetica"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16


def save_plot(fig, filename: str):
    """Saves a matplotlib figure cleanly and closes it."""
    filepath = VIZ_DIR / filename
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return filepath


def save_json(data: dict, filename: str):
    """Saves a dictionary to JSON format in the metrics folder."""
    filepath = METRICS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return filepath


def load_json(filename: str) -> dict:
    """Loads a JSON file from the metrics folder."""
    filepath = METRICS_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
