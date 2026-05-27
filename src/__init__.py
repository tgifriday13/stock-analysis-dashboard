"""
Stock Fundamentals CEO Dashboard
---------------------------------
A comprehensive stock analysis toolkit that answers the 6 core fundamental
questions driving every portfolio decision.

Usage:
    Open notebooks/main.ipynb and run all cells.
"""

from . import (
    data_fetcher,
    first_principles,
    metrics_calculator,
    threshold_manager,
    visualizations,
    panel_generator,
    report_generator,
    metric_interpreter,
    utils,
)

__version__ = "1.1.0"
__all__ = [
    "data_fetcher",
    "first_principles",
    "metrics_calculator",
    "threshold_manager",
    "visualizations",
    "panel_generator",
    "report_generator",
    "metric_interpreter",
    "utils",
]
