"""Ajusta hiperparámetros en validación y guarda los mejores por generador y algoritmo."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rl_trader.agents import ALGORITHMS  # noqa: E402
from rl_trader.experiments import load_config  # noqa: E402
from rl_trader.prices import GENERATORS  # noqa: E402
from rl_trader.tuning import best_rows, run_tuning, select_best, summarize  # noqa: E402


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("debe ser >= 1")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    p.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    p.add_argument("--algos", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    p.add_argument("--episodes", type=_positive_int, default=None)
    p.add_argument("--runs", type=_positive_int, default=None)
    p.add_argument("--jobs", type=_positive_int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tuning")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["tuning"]["n_episodes"] = args.episodes
    if args.runs is not None:
        cfg["tuning"]["n_runs"] = args.runs

    raw = run_tuning(cfg, args.generators, args.algos, args.jobs)
    summary = summarize(raw, cfg["tuning"]["grid"])
    best = select_best(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    raw.assign(params=raw["params"].apply(json.dumps)).to_csv(args.out / "tuning_raw.csv", index=False)
    summary.to_csv(args.out / "tuning_summary.csv", index=False)
    with open(args.out / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(best_rows(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
