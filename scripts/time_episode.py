"""Mide ms/episodio por generador y algoritmo y estima el tiempo total de la config."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trader.agents import ALGORITHMS  # noqa: E402
from rl_trader.discretize import fit_discretizer  # noqa: E402
from rl_trader.experiments import load_config  # noqa: E402
from rl_trader.prices import GENERATORS  # noqa: E402
from rl_trader.train import train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    p.add_argument("--episodes", type=int, default=20, help="episodios de medida")
    args = p.parse_args()
    cfg = load_config(args.config)

    print(f"{'generador':<18}{'algo':<10}{'ms/episodio':>12}")
    total = 0.0
    for gen in GENERATORS:
        disc = fit_discretizer(gen, 20, cfg["n_steps"], cfg["n_bins"])
        for algo in ALGORITHMS:
            res = train(algo, gen, disc, args.episodes, 0, cfg["n_steps"], **cfg["agent"])
            ms = res.wall_time / args.episodes * 1000
            total += ms / 1000 * cfg["n_episodes"] * cfg["n_runs"]
            print(f"{gen:<18}{algo:<10}{ms:>12.1f}")
    print(f"\nEstimación entrenamiento config ({cfg['n_episodes']} episodios x "
          f"{cfg['n_runs']} runs x {len(GENERATORS)} generadores x {len(ALGORITHMS)} algos): "
          f"{total / 60:.1f} min (sin evaluación)")


if __name__ == "__main__":
    main()
