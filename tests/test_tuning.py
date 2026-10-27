"""Tests del ajuste de hiperparámetros en miniatura."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import rl_trader.tuning as tuning
from rl_trader.experiments import run_generator
from rl_trader.prices import split_seeds
from rl_trader.tuning import expand_grid, run_tuning, select_best, summarize
from test_experiments import CFG as BASE

CFG = copy.deepcopy(BASE)
CFG["tuning"] = {"grid": {"gamma": [0.0, 0.9], "alpha": [0.05, 0.2]},
                 "n_episodes": 10, "n_runs": 2, "val_every": 5}


def test_expand_grid_determinista():
    g = {"gamma": [0.0, 0.9], "alpha": [0.05, 0.2]}
    out = expand_grid(g)
    assert out == expand_grid(dict(reversed(list(g.items()))))
    assert out[0] == {"alpha": 0.05, "gamma": 0.0}
    assert out[1] == {"alpha": 0.05, "gamma": 0.9}
    assert len(out) == 4


def test_expand_grid_clave_reservada():
    with pytest.raises(ValueError, match="score_mean"):
        expand_grid({"score_mean": [1, 2]})


def test_summarize_raw_vacio():
    with pytest.raises(ValueError, match="sin tareas"):
        summarize(pd.DataFrame(), {"x": [1]})


def test_run_tuning_filas():
    raw = run_tuning(CFG, ["browniano", "tendencia_ciclo"], ["qlearning", "sarsa"], n_jobs=1)
    assert len(raw) == 4 * 2 * 2 * 2
    assert {"generator", "algo", "params", "run", "val_final", "conv_val",
            "conv_policy", "conv_ma", "wall_time"} <= set(raw.columns)


def _raw(scores):
    rows = [{"generator": "g", "algo": "a", "params": c, "run": r, "val_final": s,
             "conv_val": None, "conv_policy": None, "conv_ma": None, "wall_time": 0.0}
            for c, ss in zip(expand_grid({"x": [1, 2, 3]}), scores) for r, s in enumerate(ss)]
    return pd.DataFrame(rows)


def test_seleccion_maximo_y_desempate():
    grid = {"x": [1, 2, 3]}
    s = summarize(_raw([[0.1, 0.1], [0.3, 0.1], [0.0, 0.2]]), grid)
    assert select_best(s)["g"]["a"] == {"x": 2}
    s = summarize(_raw([[0.2, 0.2], [0.2, 0.2], [0.1, 0.1]]), grid)
    assert select_best(s)["g"]["a"] == {"x": 1}
    assert s["n_not_conv_val"].tolist() == [2, 2, 2]


def test_solo_semillas_val(monkeypatch):
    tuning._CACHE.clear()
    usadas = []
    orig = tuning.make_series

    def espia(gen, seed, n):
        usadas.append(seed)
        return orig(gen, seed, n)

    monkeypatch.setattr(tuning, "make_series", espia)
    run_tuning(CFG, ["browniano"], ["qlearning"], n_jobs=1)
    assert usadas and set(usadas) == set(split_seeds("val", CFG["n_val_series"]))
    assert not set(usadas) & set(split_seeds("test", 1000))
    tuning._CACHE.clear()


def test_override_run_generator(tmp_path):
    ov = {"qlearning": {"alpha": 0.2, "gamma": 0.0, "epsilon_decay": 0.9}}
    s = run_generator("browniano", CFG, tmp_path, ["qlearning", "sarsa"], ov)
    q = s[s["method"] == "qlearning"].iloc[0]
    assert (q["alpha"], q["gamma"], q["epsilon_decay"]) == (0.2, 0.0, 0.9)
    sa = s[s["method"] == "sarsa"].iloc[0]
    assert (sa["alpha"], sa["gamma"], sa["epsilon_decay"]) == (0.1, 0.95, 0.99)


def _hp(s, method):
    r = s[s["method"] == method].iloc[0]
    return (r["alpha"], r["gamma"], r["epsilon_decay"])


def test_override_parcial_algo_ausente(tmp_path):
    ov = {"sarsa": {"alpha": 0.3, "gamma": 0.5}}
    s = run_generator("browniano", CFG, tmp_path, ["qlearning", "sarsa"], ov)
    assert _hp(s, "sarsa") == (0.3, 0.5, 0.99)
    assert _hp(s, "qlearning") == (0.1, 0.95, 0.99)


@pytest.mark.parametrize("ov", [None, {}])
def test_override_generador_ausente_usa_default(tmp_path, ov):
    s = run_generator("browniano", CFG, tmp_path, ["qlearning"], ov)
    assert _hp(s, "qlearning") == (0.1, 0.95, 0.99)


def test_override_no_muta_cfg(tmp_path):
    antes = copy.deepcopy(CFG)
    ov = {"qlearning": {"alpha": 0.2}}
    ov_antes = copy.deepcopy(ov)
    run_generator("browniano", CFG, tmp_path, ["qlearning"], ov)
    assert CFG == antes
    assert ov == ov_antes


def test_empate_exacto_primer_combo_orden_grid():
    grid = {"b": [1, 2], "a": [10, 20]}
    combos = expand_grid(grid)
    rows = [{"generator": "g", "algo": "x", "params": c, "run": r, "val_final": 0.5,
             "conv_val": None, "conv_policy": None, "conv_ma": None, "wall_time": 0.0}
            for c in combos for r in range(2)]
    best = select_best(summarize(pd.DataFrame(rows), grid))
    assert best["g"]["x"] == combos[0] == {"a": 10, "b": 1}


def test_run_tuning_determinista():
    cols = ["generator", "algo", "params", "run", "val_final", "conv_val",
            "conv_policy", "conv_ma"]
    tuning._CACHE.clear()
    a = run_tuning(CFG, ["browniano"], ["qlearning", "sarsa"], n_jobs=1)[cols]
    tuning._CACHE.clear()
    b = run_tuning(CFG, ["browniano"], ["qlearning", "sarsa"], n_jobs=1)[cols]
    pd.testing.assert_frame_equal(a, b)
    tuning._CACHE.clear()


def test_cli_tune_genera_ficheros(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(CFG), encoding="utf-8")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "tune.py"
    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path),
                        "--generators", "browniano", "--algos", "qlearning",
                        "--episodes", "5", "--runs", "1", "--jobs", "1",
                        "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    for f in ("tuning_raw.csv", "tuning_summary.csv", "best_params.json"):
        assert (out / f).exists()
    best = json.loads((out / "best_params.json").read_text(encoding="utf-8"))
    assert set(best["browniano"]["qlearning"]) == {"alpha", "gamma"}
