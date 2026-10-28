"""Extensiones: nº de tramos, comisión, Monte Carlo y entrenamiento mezclando generadores."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rl_trader.experiments import load_config, run_generator, run_mixed  # noqa: E402
from rl_trader.prices import GENERATORS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["bins", "cost", "mc", "mixed"])
    p.add_argument("--value", type=float, default=None,
                   help="nº de tramos (bins) o comisión (cost); no aplica a mc ni mixed")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    p.add_argument("--params", type=Path, default=ROOT / "results" / "tuning" / "sweep_best.json")
    p.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "ext")
    args = p.parse_args()

    cfg = load_config(args.config)
    best = json.loads(args.params.read_text(encoding="utf-8"))
    ext = cfg["ext"]
    tag = args.kind if args.value is None else f"{args.kind}{args.value:g}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    if args.kind == "mixed":
        summary = run_mixed(cfg, out, ["qlearning", "sarsa"], ext["mixed_episodes"])
        summary.to_csv(out / "summary.csv", index=False)
        return

    if args.kind == "bins":
        cfg["n_bins"] = int(args.value)
    elif args.kind == "cost":
        cfg["cost_rate"] = args.value
    tables = []
    for g in args.generators:
        if args.kind == "mc":
            # Sin ajuste propio: se reutilizan los parámetros de Q-learning (α no se usa en MC)
            base = {k: v for k, v in best[g]["qlearning"].items() if k != "n_episodes"}
            overrides = {a: {**base, "n_episodes": ext["mc_episodes"]} for a in ("mc", "mc_off")}
            algos = ["mc", "mc_off"]
        else:
            overrides, algos = best[g], ["qlearning", "sarsa"]
        t = run_generator(g, cfg, out, algos, overrides)
        t.to_csv(out / f"summary_{g}.csv", index=False)
        tables.append(t)
    pd.concat(tables, ignore_index=True).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
