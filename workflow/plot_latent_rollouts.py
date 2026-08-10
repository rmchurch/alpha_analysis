#!/usr/bin/env python3
"""Plot free-rollout and persistence latent error by forecast horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    horizons = sorted(int(key) for key in metrics["latent_mse_by_horizon"])
    rollout = [metrics["latent_mse_by_horizon"][str(key)] for key in horizons]
    persistence = [metrics["persistence_mse_by_horizon"][str(key)] for key in horizons]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(horizons, rollout, marker="o", label="free rollout")
    axis.plot(horizons, persistence, marker="o", label="persistence")
    axis.set(xlabel="forecast horizon", ylabel="latent MSE", yscale="log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
