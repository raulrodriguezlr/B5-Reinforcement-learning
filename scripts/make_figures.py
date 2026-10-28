"""Genera figuras (PNG 300 dpi + PDF) y tabla resumen desde results/final."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rl_trader.figures import build_ext_tables, make_all, table_to_markdown  # noqa: E402


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
                      random_seed=cfg["random_base_seed"], n_runs=cfg["n_runs"],
                      ma_window=cfg["convergence"]["ma_window"],
                      ma_tol_frac=cfg["convergence"]["ma_tol_frac"]):
        print(f)
    ext_dir = args.results.parent / "ext"
    keys = {"bins5": "bins5", "bins20": "bins20", "cost0": "cost0", "cost0.005": "cost0.005",
            "mc": "mc", "mixed": "mixed"}
    if all((ext_dir / d / "summary.csv").is_file() for d in keys.values()):
        ext = {k: pd.read_csv(ext_dir / d / "summary.csv") for k, d in keys.items()}
        tables = build_ext_tables(pd.read_csv(args.results / "summary.csv"), ext)
        for name, t in tables.items():
            t.to_csv(args.out / f"tabla_ext_{name}.csv", index=False)
            (args.out / f"tabla_ext_{name}.md").write_text(table_to_markdown(t), encoding="utf-8")
            print(args.out / f"tabla_ext_{name}.csv")


if __name__ == "__main__":
    main()
