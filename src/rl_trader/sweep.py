"""Barrido de epsilon_min y del número de episodios sobre validación (nunca test)."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rl_trader.tuning import _prepare
from rl_trader.train import train, val_convergence_episode


def run_sweep_task(task: tuple) -> list[dict]:
    """Entrena (generator, algo, params, epsilon_min, run, cfg) y devuelve la curva de validación."""
    generator, algo, params, eps_min, run, cfg = task
    scfg = cfg["sweep"]
    disc, val = _prepare(generator, cfg)
    conv = {**cfg["convergence"], "val_every": scfg["val_every"]}
    res = train(algo, generator, disc, scfg["n_episodes"], run, cfg["n_steps"],
                val_series=val, **{**cfg["agent"], **params, "epsilon_min": eps_min}, **conv)
    online = float(res.episode_rewards[-scfg["online_window"]:].mean())
    return [{"generator": generator, "algo": algo, "epsilon_min": eps_min, "run": run,
             "episode": int(e) + 1, "val": float(v), "online_reward": online,
             "wall_time": res.wall_time}
            for e, v in zip(res.val_episodes, res.val_median_log_return)]


def run_sweep(cfg: dict, best_params: dict, generators: list[str], algos: list[str],
              n_jobs: int = 1) -> pd.DataFrame:
    """Una fila por (generador, algoritmo, epsilon_min, run, punto de validación)."""
    scfg = cfg["sweep"]
    tasks = [(g, a, best_params[g][a], em, r, cfg) for g in generators for a in algos
             for em in scfg["epsilon_min"] for r in range(scfg["n_runs"])]
    if n_jobs <= 1:
        chunks = [run_sweep_task(t) for t in tasks]
    else:
        with Pool(n_jobs) as pool:
            chunks = pool.map(run_sweep_task, tasks, chunksize=1)
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def select_sweep(raw: pd.DataFrame, default_eps_min: float = 0.05, plateau: int = 3,
                 tol_abs: float = 0.01, tol_rel: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Elige epsilon_min y nº de episodios por (generador, algoritmo).

    Meseta = media de los últimos `plateau` puntos de la curva media. Entre los epsilon_min a menos de
    tolerancia del mejor se prefiere el por defecto; N es el primer punto desde el que la curva queda en tolerancia.
    """
    rows, best = [], {}
    for (g, a), sub in raw.groupby(["generator", "algo"], sort=False):
        curves = sub.groupby(["epsilon_min", "episode"])["val"].mean().unstack(0)
        online = sub.groupby("epsilon_min")["online_reward"].mean()
        plat = curves.iloc[-plateau:].mean()
        tol = max(tol_abs, tol_rel * abs(plat.max()))
        near = plat[plat >= plat.max() - tol]
        chosen = default_eps_min if default_eps_min in near.index else float(near.idxmax())
        curve = curves[chosen]
        episodes = curve.index.to_numpy()
        n_total = int(episodes[-1])
        ep0 = val_convergence_episode(episodes - 1, curve.to_numpy(), n_total, tol_abs, tol_rel, 0)
        n_star = int(ep0) + 1
        for em in curves.columns:
            rows.append({"generator": g, "algo": a, "epsilon_min": float(em),
                         "val_plateau": float(plat[em]), "online_reward_last": float(online[em]),
                         "chosen": bool(em == chosen)})
        best.setdefault(g, {})[a] = {"epsilon_min": float(chosen), "n_episodes": n_star}
    return pd.DataFrame(rows), best
