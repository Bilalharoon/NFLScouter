"""
src/visualization.py
--------------------
Visualization module for the NFL Scouting Engine.

Produces a Seaborn scatter plot of:
  X = success_rate_scaled
  Y = explosiveness_scaled
  size = CSS_Score (scaled to marker area)
  hue = position

Top-N players by CSS_Score are annotated with their name.
Plot is saved to the specified output path.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_POSITION_PALETTE = {"WR": "#4C72B0", "RB": "#DD8452", "TE": "#55A868"}
_TOP_N_ANNOTATE = 10
_SIZE_SCALE = 600  # CSS_Score * _SIZE_SCALE → marker area in points²


def plot_css(
    scored: pd.DataFrame,
    output_path: str = "outputs/css_scatter.png",
    top_n: int = _TOP_N_ANNOTATE,
) -> None:
    """
    Render and save the CSS scatter plot.

    Parameters
    ----------
    scored : pd.DataFrame
        Output of ``scoring.compute_css()``.
    output_path : str
        File path for the saved PNG (directories created if needed).
    top_n : int
        Number of top-ranked players to annotate by name.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3D4F")

    ax.tick_params(colors="#C0C3D0")
    ax.xaxis.label.set_color("#C0C3D0")
    ax.yaxis.label.set_color("#C0C3D0")
    ax.title.set_color("#FFFFFF")

    # Plot each position group
    for position, group in scored.groupby("position"):
        color = _POSITION_PALETTE.get(position, "#AAAAAA")
        ax.scatter(
            group["success_rate_scaled"],
            group["explosiveness_scaled"],
            s=group["CSS_Score"] * _SIZE_SCALE,
            c=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=position,
            zorder=3,
        )

    # Annotate top-N players
    top_players = scored.nlargest(top_n, "CSS_Score")
    for _, row in top_players.iterrows():
        ax.annotate(
            row["player_name"],
            xy=(row["success_rate_scaled"], row["explosiveness_scaled"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#FFFFFF",
            alpha=0.90,
            zorder=4,
        )

    # Reference lines at medians
    ax.axvline(scored["success_rate_scaled"].median(), color="#555870", lw=1, ls="--", alpha=0.6)
    ax.axhline(scored["explosiveness_scaled"].median(), color="#555870", lw=1, ls="--", alpha=0.6)

    ax.set_xlabel("Success Rate (scaled)", fontsize=12, labelpad=8)
    ax.set_ylabel("Explosiveness (scaled)", fontsize=12, labelpad=8)
    ax.set_title(
        "NFL Rookie Scouting — Composite Scout Score (CSS)\n"
        "Size = CSS Score  |  Dashed lines = median",
        fontsize=14,
        pad=14,
        color="#FFFFFF",
    )

    legend = ax.legend(
        title="Position",
        title_fontsize=9,
        fontsize=9,
        framealpha=0.2,
        facecolor="#2A2D3A",
        edgecolor="#3A3D4F",
        labelcolor="#C0C3D0",
    )
    legend.get_title().set_color("#C0C3D0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)
