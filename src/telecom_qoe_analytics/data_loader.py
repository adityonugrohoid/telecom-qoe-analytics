"""
Data loader and ML utilities for QOE analytics.
Provides consistent data loading, preprocessing, and helper functions.
Refactored to support strict package layout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "final"

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_parquet(name: str) -> pd.DataFrame:
    """Load a single parquet file from data/final/."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_parquet(path)


def load_all() -> dict[str, pd.DataFrame]:
    """Load all core datasets."""
    return {
        "users": load_parquet("users"),
        "cells": load_parquet("cells"),
        "sessions": load_parquet("sessions"),
    }


def get_merged_dataset(
    include_users: bool = True,
    include_cells: bool = True,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load sessions and merge with users/cells for enriched analysis.
    
    Args:
        include_users: Merge user attributes
        include_cells: Merge cell attributes  
        sample_frac: Optional fraction to sample (for large data)
        random_state: Random seed for sampling
    """
    df = load_parquet("sessions")
    
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
    
    if include_users:
        users = load_parquet("users")
        df = df.merge(users, on="user_id", how="left", suffixes=("", "_user"))
    
    if include_cells:
        cells = load_parquet("cells")
        df = df.merge(cells, on="cell_id", how="left", suffixes=("", "_cell"))
        
        # Derive network_type from band if available
        if "band" in df.columns:
            # Simple heuristic: 'n' prefix usually denotes 5G NR (e.g., n78), 'L' or others LTE (4G)
            # Adjust based on actual data patterns (e.g., L2100 -> 4G)
            df["network_type"] = np.where(df["band"].str.startswith("n"), "5G", "4G")
    
    return df


# -----------------------------------------------------------------------------
# Time-Aware Splits for ML
# -----------------------------------------------------------------------------

def get_time_split(
    df: pd.DataFrame,
    time_col: str = "timestamp_start",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no leakage) into train/val/test.
    
    Args:
        df: DataFrame with timestamp column
        time_col: Name of timestamp column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (remainder goes to test)
    
    Returns:
        (train_df, val_df, test_df)
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return (
        df_sorted.iloc[:train_end],
        df_sorted.iloc[train_end:val_end],
        df_sorted.iloc[val_end:],
    )


# -----------------------------------------------------------------------------
# Target Engineering for Classification
# -----------------------------------------------------------------------------

def create_qoe_targets(
    df: pd.DataFrame,
    qoe_col: str = "qoe_score",
    low_percentile: float = 10,
    tier_bins: int = 5,
) -> pd.DataFrame:
    """
    Create classification targets from QoE score.
    
    Adds columns:
        - low_qoe: Binary (1 if in bottom percentile)
        - qoe_tier: Multi-class (quintile labels 1-5)
    """
    df = df.copy()
    
    # Binary: bottom decile
    threshold = np.percentile(df[qoe_col].dropna(), low_percentile)
    df["low_qoe"] = (df[qoe_col] <= threshold).astype(int)
    
    # Multi-class: quintiles
    df["qoe_tier"] = pd.qcut(
        df[qoe_col], q=tier_bins, labels=range(1, tier_bins + 1)
    ).astype(int)
    
    return df


# -----------------------------------------------------------------------------
# Feature Engineering Helpers
# -----------------------------------------------------------------------------

def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "timestamp_start",
) -> pd.DataFrame:
    """Add hour, day-of-week, weekend, and period features."""
    df = df.copy()
    ts = pd.to_datetime(df[time_col])
    
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["period"] = pd.cut(
        ts.dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    )
    
    return df


def get_numeric_features(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Get list of numeric feature columns, excluding specified columns."""
    exclude = exclude or []
    exclude_set = set(exclude + ["session_id", "user_id", "cell_id", "event_id"])
    
    return [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_set
    ]


def get_categorical_features(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Get list of categorical feature columns."""
    exclude = exclude or []
    exclude_set = set(exclude + ["session_id", "user_id", "cell_id", "event_id"])
    
    return [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in exclude_set
    ]


# -----------------------------------------------------------------------------
# Plotting Setup
# -----------------------------------------------------------------------------

def setup_plotting(
    style: str = "ticks",
    context: str = "notebook",
    palette: str = "deep",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Configure consistent, premium Seaborn/Matplotlib styling.
    Uses modern defaults: 'ticks' style, 'magma' palette, and refined typography.
    """
    # Set the base theme
    sns.set_theme(style=style, context=context, palette=palette)
    
    print(f"📈 Plotting environment set: style={style}, palette={palette}, context={context}")


def save_figure(name: str, fig: plt.Figure | None = None, dpi: int = 150) -> Path:
    """Save current figure to reports/ directory."""
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    path = reports_dir / f"{name}.png"
    if fig is None:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    
    return path


# -----------------------------------------------------------------------------
# Quick Stats
# -----------------------------------------------------------------------------

def quick_stats(df: pd.DataFrame, target_col: str = "qoe_score") -> pd.DataFrame:
    """Generate quick summary statistics for the target variable."""
    return df[target_col].describe().to_frame().T


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate missing value report."""
    missing = df.isnull().sum()
    percent = 100 * missing / len(df)
    
    return pd.DataFrame({
        "missing_count": missing,
        "missing_pct": percent.round(2),
        "dtype": df.dtypes,
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)


# -----------------------------------------------------------------------------
# Schema Validation
# -----------------------------------------------------------------------------

def validate_datasets(datasets: dict[str, pd.DataFrame] | None = None) -> dict[str, bool]:
    """Validate all datasets against the schema."""
    from .schema import validate_table
    
    if datasets is None:
        datasets = load_all()
        
    results = {}
    for name, df in datasets.items():
        try:
            results[name] = validate_table(df, name)
        except Exception as e:
            print(f"Validation failed for {name}: {e}")
            results[name] = False
            
    return results
