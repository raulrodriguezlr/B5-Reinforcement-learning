"""Genera figuras (PNG 300 dpi + PDF) y tabla resumen desde results/final."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trader.figures import make_all  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results" / "final")
    p.add_argument("--out", type=Path, default=ROOT / "report" / "figures")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    args = p.parse_args(argv)
    if not args.results.is_dir() or not (args.results / "summary.csv").is_file():
        p.error(f"no se encuentran resultados en {args.results}: "
                "ejecutar antes scripts/run_experiments.py")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for f in make_all(args.results, args.out, n_steps=cfg["n_steps"],
                      random_seed=cfg["random_base_seed"], n_runs=cfg["n_runs"]):
        print(f)


if __name__ == "__main__":
    main()
