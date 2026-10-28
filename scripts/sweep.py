"""Barre epsilon_min y nº de episodios en validación y guarda la elección por generador y algoritmo."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trader.agents import ALGORITHMS  # noqa: E402
from rl_trader.experiments import load_config  # noqa: E402
from rl_trader.prices import GENERATORS  # noqa: E402
from rl_trader.sweep import run_sweep, select_sweep  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    p.add_argument("--params", type=Path, default=ROOT / "results" / "tuning" / "best_params.json",
                   help="best_params.json de tune.py: punto de partida del barrido")
    p.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    p.add_argument("--algos", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tuning")
    args = p.parse_args()

    cfg = load_config(args.config)
    best_params = json.loads(args.params.read_text(encoding="utf-8"))
    raw = run_sweep(cfg, best_params, args.generators, args.algos, args.jobs)
    summary, choice = select_sweep(raw, cfg["agent"]["epsilon_min"])
    merged = {g: {a: {**best_params[g][a], **c} for a, c in algs.items()}
              for g, algs in choice.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / "sweep_raw.csv", index=False)
    summary.to_csv(args.out / "sweep_summary.csv", index=False)
    (args.out / "sweep_best.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
