"""Tests de la orquestación de experimentos en miniatura."""
import copy
from pathlib import Path

import pandas as pd
import pytest

from rl_trader.experiments import load_config, run_generator

CFG = {
    "n_steps": 100,
    "n_bins": 4,
    "n_fit_series": 5,
    "n_episodes": 20,
    "n_runs": 1,
    "n_test_series": 2,
    "n_val_series": 2,
    "agent": {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_min": 0.05,
              "epsilon_decay": 0.99, "alpha_decay": 0.99, "alpha_min": 0.01},
    "convergence": {"policy_window": 5, "policy_agreement": 0.95, "ma_window": 5,
                    "ma_tol_frac": 0.1, "val_every": 10, "val_tol_abs": 0.01,
                    "val_tol_rel": 0.05, "val_window": 5},
    "random_base_seed": 0,
}


def test_run_generator_miniatura(tmp_path):
    summary = run_generator("browniano", CFG, tmp_path, ["qlearning", "sarsa"])
    d = tmp_path / "browniano"
    esperados = ["discretizer.json", "train_qlearning_run0.npz", "train_sarsa_run0.npz",
                 "metrics_qlearning.csv", "metrics_sarsa.csv",
                 "policy_qlearning.csv", "policy_sarsa.csv",
                 "metrics_random.csv", "metrics_buy_and_hold.csv"]
    for f in esperados:
        assert (d / f).exists(), f
    assert list(summary["method"]) == ["qlearning", "sarsa", "random", "buy_and_hold"]
    assert (summary["generator"] == "browniano").all()
    m = pd.read_csv(d / "metrics_qlearning.csv")
    assert len(m) == 2 and "run" in m.columns
    assert len(pd.read_csv(d / "policy_qlearning.csv")) == 4 * 2


def test_run_generator_determinista(tmp_path):
    a = run_generator("browniano", CFG, tmp_path / "a", ["qlearning"])
    b = run_generator("browniano", CFG, tmp_path / "b", ["qlearning"])
    cols = [c for c in a.columns
            if not c.startswith(("wall_time", "conv_time", "conv_val_time", "val_time"))]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_n_val_series_cero(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["n_val_series"] = 0
    with pytest.raises(ValueError):
        run_generator("browniano", cfg, tmp_path, ["qlearning"])


def test_load_config_default():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "default.json")
    assert cfg["n_steps"] == 500 and cfg["n_val_series"] >= 1
