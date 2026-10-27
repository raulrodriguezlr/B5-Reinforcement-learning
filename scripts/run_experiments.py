"""Lanza los experimentos para los generadores y algoritmos elegidos."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rl_trader.agents import ALGORITHMS  # noqa: E402
from rl_trader.experiments import load_config, run_generator  # noqa: E402
from rl_trader.prices import GENERATORS  # noqa: E402


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
    p.add_argument("--test-series", type=_positive_int, default=None)
    p.add_argument("--val-series", type=_positive_int, default=None)
    p.add_argument("--params", type=Path, default=None,
                   help="best_params.json de tune.py: {generator: {algo: params}}")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["n_episodes"] = args.episodes
    if args.runs is not None:
        cfg["n_runs"] = args.runs
    if args.test_series is not None:
        cfg["n_test_series"] = args.test_series
    if args.val_series is not None:
        cfg["n_val_series"] = args.val_series

    overrides = {}
    if args.params is not None:
        with open(args.params, encoding="utf-8") as f:
            overrides = json.load(f)

    args.out.mkdir(parents=True, exist_ok=True)
    tables = [run_generator(g, cfg, args.out, args.algos, overrides.get(g))
              for g in args.generators]
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)

    metrics = ["final_balance", "log_return", "excess_return_pct", "sharpe",
               "n_trades", "time_in_market"]
    cols = ["generator", "method"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_median", f"{m}_std_runs"]
    cols += ["conv_episode_policy_mean", "conv_episode_ma_mean",
             "conv_time_policy_mean", "conv_val_mean", "conv_val_time_mean",
             "n_not_converged_val", "wall_time_mean", "val_time_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary[[c for c in cols if c in summary.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
