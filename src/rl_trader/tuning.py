"""Ajuste de hiperparámetros sobre series de validación (nunca test)."""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rl_trader.discretize import fit_discretizer
from rl_trader.prices import make_series, split_seeds
from rl_trader.train import train

# Caché por proceso: (discretizador, series val) por generador y configuración
_CACHE: dict[tuple, tuple] = {}

_NON_PARAM = ("generator", "algo", "combo_idx", "score_mean", "score_std",
              "conv_val_mean", "conv_policy_mean", "n_not_conv_val")


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Combos del grid en orden determinista (claves ordenadas, product)."""
    for k in grid:
        if k in _NON_PARAM:
            raise ValueError(f"clave de grid reservada: {k}")
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _prepare(generator: str, cfg: dict) -> tuple:
    key = (generator, cfg["n_steps"], cfg["n_bins"], cfg["n_fit_series"], cfg["n_val_series"])
    if key not in _CACHE:
        disc = fit_discretizer(generator, cfg["n_fit_series"], cfg["n_steps"], cfg["n_bins"])
        val = [make_series(generator, sd, cfg["n_steps"])
               for sd in split_seeds("val", cfg["n_val_series"])]
        _CACHE[key] = (disc, val)
    return _CACHE[key]


def run_task(task: tuple) -> dict:
    """Entrena una tarea (generator, algo, combo, run, cfg) y devuelve su fila de resultados."""
    generator, algo, combo, run, cfg = task
    tcfg = cfg["tuning"]
    disc, val = _prepare(generator, cfg)
    conv = {**cfg["convergence"], "val_every": tcfg["val_every"]}
    res = train(algo, generator, disc, tcfg["n_episodes"], run, cfg["n_steps"],
                val_series=val, **{**cfg["agent"], **combo}, **conv)
    vals = res.val_median_log_return
    return {"generator": generator, "algo": algo, "params": combo, "run": run,
            "val_final": float(vals[-1]) if len(vals) > 0 else float("nan"),
            "conv_val": res.convergence_episode_val,
            "conv_policy": res.convergence_episode_policy,
            "conv_ma": res.convergence_episode_ma,
            "wall_time": res.wall_time}


def run_tuning(cfg: dict, generators: list[str], algos: list[str],
               n_jobs: int = 1) -> pd.DataFrame:
    """Ejecuta todas las tareas (generador, algoritmo, combo, run); n_jobs=1 sin pool."""
    combos = expand_grid(cfg["tuning"]["grid"])
    tasks = [(g, a, c, r, cfg) for g in generators for a in algos
             for c in combos for r in range(cfg["tuning"]["n_runs"])]
    if n_jobs <= 1:
        rows = [run_task(t) for t in tasks]
    else:
        with Pool(n_jobs) as pool:
            rows = pool.map(run_task, tasks, chunksize=1)
    return pd.DataFrame(rows)


def _mean_or_nan(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


def summarize(raw: pd.DataFrame, grid: dict[str, list]) -> pd.DataFrame:
    """Una fila por (generator, algo, combo) con score medio y std entre runs."""
    if raw.empty:
        raise ValueError("tuning sin tareas: revisar generadores/algos")
    combos = expand_grid(grid)
    rows = []
    for (g, a), sub in raw.groupby(["generator", "algo"], sort=False):
        for i, c in enumerate(combos):
            d = sub[sub["params"].apply(lambda p: p == c)]
            if d.empty:
                continue
            rows.append({"generator": g, "algo": a, "combo_idx": i, **c,
                         "score_mean": float(d["val_final"].mean()),
                         "score_std": float(d["val_final"].std(ddof=1)) if len(d) > 1 else 0.0,
                         "conv_val_mean": _mean_or_nan(d["conv_val"]),
                         "conv_policy_mean": _mean_or_nan(d["conv_policy"]),
                         "n_not_conv_val": int(d["conv_val"].isna().sum())})
    return pd.DataFrame(rows)


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Fila ganadora por (generator, algo): max score_mean, empate al primer combo."""
    out = []
    for _, sub in summary.groupby(["generator", "algo"], sort=False):
        sub = sub.sort_values("combo_idx")
        out.append(sub.loc[sub["score_mean"].fillna(-np.inf).idxmax()])  # idxmax: primero
    return pd.DataFrame(out)


def select_best(summary: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Mejores params como {generator: {algo: params}}."""
    param_cols = [c for c in summary.columns if c not in _NON_PARAM]
    best: dict[str, dict[str, dict]] = {}
    for _, row in best_rows(summary).iterrows():
        best.setdefault(row["generator"], {})[row["algo"]] = {
            c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in param_cols}
    return best
