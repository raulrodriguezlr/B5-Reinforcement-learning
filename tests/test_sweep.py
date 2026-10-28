"""Tests del barrido de epsilon_min y del número de episodios."""
import copy

import numpy as np
import pandas as pd

from rl_trader.sweep import run_sweep, select_sweep
from test_experiments import CFG as BASE

CFG = copy.deepcopy(BASE)
CFG["sweep"] = {"epsilon_min": [0.0, 0.2], "n_episodes": 10, "n_runs": 2,
                "val_every": 5, "online_window": 4}


def _raw(curves: dict[float, list[float]]) -> pd.DataFrame:
    rows = [{"generator": "g", "algo": "a", "epsilon_min": em, "run": run,
             "episode": 100 * (i + 1), "val": v, "online_reward": 0.0, "wall_time": 0.0}
            for em, vals in curves.items() for run in range(2) for i, v in enumerate(vals)]
    return pd.DataFrame(rows)


def test_run_sweep_filas():
    best = {"browniano": {"qlearning": {"alpha": 0.1, "gamma": 0.9, "epsilon_decay": 0.9}}}
    raw = run_sweep(CFG, best, ["browniano"], ["qlearning"], n_jobs=1)
    assert len(raw) == 2 * 2 * 2  # epsilon_min x runs x puntos de validación
    assert set(raw["episode"]) == {5, 10}
    assert set(raw["epsilon_min"]) == {0.0, 0.2}


def test_select_n_primer_punto_en_meseta():
    raw = _raw({0.05: [0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 0.2: [0.0, 0.1, 0.2, 0.2, 0.2, 0.2]})
    _, best = select_sweep(raw, default_eps_min=0.05)
    assert best["g"]["a"] == {"epsilon_min": 0.05, "n_episodes": 300}


def test_select_prefiere_default_en_empate():
    raw = _raw({0.0: [1.0] * 6, 0.05: [1.0] * 6, 0.2: [0.0] * 6})
    summary, best = select_sweep(raw, default_eps_min=0.05)
    assert best["g"]["a"]["epsilon_min"] == 0.05
    assert summary["chosen"].sum() == 1


def test_select_cambia_si_default_queda_fuera_de_tolerancia():
    raw = _raw({0.05: [0.0] * 6, 0.2: [1.0] * 6})
    _, best = select_sweep(raw, default_eps_min=0.05)
    assert best["g"]["a"]["epsilon_min"] == 0.2
    assert best["g"]["a"]["n_episodes"] == 100
