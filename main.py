"""
main.py
-------
CLI entry point for the NFL Scouting Engine.

Usage:
    python main.py [--seasons 2023 2024 2025] [--min-snaps 50]
                   [--weights 0.40 0.35 0.25] [--output outputs/css_scatter.png]

Example:
    python main.py --seasons 2023 2024 --min-snaps 75 --weights 0.5 0.3 0.2
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.pipeline import run, DEFAULT_SEASONS, DEFAULT_MIN_SNAPS, DEFAULT_OUTPUT_PATH
from src.scoring import METRIC_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DISPLAY_COLS = [
    "player_name", "position", "season", "total_snaps",
    "success_rate", "explosiveness", "reliability", "CSS_Score", "tier",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NFL Rookie Scouting Engine — Composite Scout Score (CSS)"
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
        help="NFL seasons to include (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--min-snaps", type=int, default=DEFAULT_MIN_SNAPS,
        help="Minimum offensive snaps to qualify (default: 50)",
    )
    parser.add_argument(
        "--weights", nargs=3, type=float, default=None,
        metavar=("W_SUCCESS", "W_EXPLOSIVENESS", "W_RELIABILITY"),
        help="CSS weights for success_rate, explosiveness, reliability (must sum to 1.0)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for scatter plot PNG (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    weights = None
    if args.weights is not None:
        weights = dict(zip(METRIC_COLS, args.weights))

    css_df = run(
        seasons=args.seasons,
        min_snaps=args.min_snaps,
        weights=weights,
        output_path=args.output,
    )

    css_df.to_csv('outputs/css_df.csv')
    print('DF saved to CSV')

    # Pretty-print ranked results
    display = css_df.reset_index()[["rank"] + DISPLAY_COLS].copy()
    display["CSS_Score"] = display["CSS_Score"].round(4)
    display["success_rate"] = display["success_rate"].round(3)
    display["explosiveness"] = display["explosiveness"].round(3)
    display["reliability"] = display["reliability"].round(3)

    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n" + "=" * 80)
    print("  NFL ROOKIE COMPOSITE SCOUT SCORE (CSS) — TOP RANKED PROSPECTS")
    print("=" * 80)
    print(display.to_string(index=False))
    print(f"\n📊  Scatter plot saved → {args.output}")


if __name__ == "__main__":
    main()
