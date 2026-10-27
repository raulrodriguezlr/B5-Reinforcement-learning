"""Tests de figuras y tabla con datos sintéticos (2 bins, pocos episodios)."""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from rl_trader import figures as F  # noqa: E402
from scripts_loader import load_make_figures  # noqa: E402

N_BINS = 2
N_RUNS = 2
N_SERIES = 3


def _make_results(root):
    rng = np.random.default_rng(0)
    rows = []
    for g in F.GENERATORS_ORDER:
        d = root / g
        d.mkdir(parents=True)
        (d / "discretizer.json").write_text(
            json.dumps({"n_bins": N_BINS, "edges": [0.0]}), encoding="utf-8")
        for algo in F.ALGO_ORDER:
            pol = []
            for run in range(N_RUNS):
                Q = rng.normal(size=(N_BINS, 2, 3))
                np.savez_compressed(d / f"train_{algo}_run{run}.npz", Q=Q,
                                    val_episodes=np.arange(0, 20, 5),
                                    val_median_log_return=rng.normal(size=4))
                for b in range(N_BINS):
                    for pos in (0, 1):
                        pol.append({"algo": algo, "run": run, "bin": b, "pos": pos,
                                    "target_position": int(rng.integers(2))})
            pd.DataFrame(pol).to_csv(d / f"policy_{algo}.csv", index=False)
            pd.DataFrame({"run": np.repeat(range(N_RUNS), N_SERIES),
                          "log_return": rng.normal(size=N_RUNS * N_SERIES)}
                         ).to_csv(d / f"metrics_{algo}.csv", index=False)
        for m in ("buy_and_hold", "random"):
            pd.DataFrame({"log_return": rng.normal(size=N_SERIES)}).to_csv(
                d / f"metrics_{m}.csv", index=False)
        for m in F.METHOD_ORDER:
            is_agent = m in F.ALGO_ORDER
            rows.append({
                "generator": g, "method": m, "final_balance_median": 1000.0,
                "log_return_mean": 0.1, "log_return_std_runs": 0.02 if is_agent else np.nan,
                "excess_return_pct_median": 1.0, "sharpe_median": 0.1,
                "max_drawdown_median": 0.3, "n_trades_median": 10.0,
                "time_in_market_median": 0.5,
                "conv_val_mean": np.nan if m == "qlearning" else 100.0 if is_agent else np.nan,
                "n_not_converged_val": 2 if m == "qlearning" else 0 if is_agent else np.nan,
                "wall_time_mean": 5.0 if is_agent else np.nan,
            })
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)


@pytest.fixture()
def results(tmp_path):
    root = tmp_path / "final"
    _make_results(root)
    return root


def test_fig_episodio(results):
    data = {g: F.load_episode_data(results, g, n_steps=60) for g in F.GENERATORS_ORDER}
    fig = F.fig_episodio(data)
    assert isinstance(fig, Figure) and len(fig.axes) == 8
    plt.close(fig)


def test_fig_politica(results):
    fig = F.fig_politica(F.load_policy_data(results))
    assert isinstance(fig, Figure) and len(fig.axes) == 9  # 8 paneles + barra de color
    plt.close(fig)


def test_fig_aprendizaje(results):
    fig = F.fig_aprendizaje(F.load_learning_data(results))
    assert isinstance(fig, Figure) and len(fig.axes) == 4
    plt.close(fig)


def test_fig_test(results):
    fig = F.fig_test(F.load_test_data(results))
    assert isinstance(fig, Figure) and len(fig.axes) == 4
    plt.close(fig)


def test_tabla_columnas_y_nc(results):
    t = F.build_table(pd.read_csv(results / "summary.csv"), n_runs=N_RUNS)
    assert list(t.columns) == F.TABLE_COLUMNS
    assert len(t) == 16
    q = t[(t["Generador"] == "Browniano") & (t["Método"] == "Q-learning")].iloc[0]
    assert q["Conv. val (episodio)"] == "n/c (0/2)"
    sa = t[(t["Generador"] == "Browniano") & (t["Método"] == "Sarsa")].iloc[0]
    assert sa["Conv. val (episodio)"] == "100 (2/2)"
    bh = t[(t["Método"] == "Buy & hold")].iloc[0]
    assert bh["Tiempo entren. (s)"] == "n/c"
    assert bh["Conv. val (episodio)"] == "n/c"


def test_cli_genera_ficheros(results, tmp_path):
    # El CLI pasa por make_all; los episodios usan 500 pasos (rápido con 2 bins)
    out = tmp_path / "out"
    load_make_figures().main(["--results", str(results), "--out", str(out)])
    names = {p.name for p in out.iterdir()}
    for n in ("fig_episodio", "fig_politica", "fig_aprendizaje", "fig_test"):
        assert f"{n}.png" in names and f"{n}.pdf" in names
    assert {"tabla_resumen.csv", "tabla_resumen.md"} <= names


def test_cli_sin_resultados(tmp_path):
    # Resultados ausentes: error de argparse (exit 2), sin traceback
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    r = subprocess.run([sys.executable, str(script), "--results", str(tmp_path),
                        "--out", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "no se encuentran resultados" in r.stderr
    assert "Traceback" not in r.stderr


def _summary_row(**kw):
    base = {"generator": "browniano", "method": "qlearning", "final_balance_median": 1234.5678,
            "log_return_mean": 0.1, "log_return_std_runs": 0.02,
            "excess_return_pct_median": 1.0, "sharpe_median": 0.1,
            "max_drawdown_median": 0.3, "n_trades_median": 10.0,
            "time_in_market_median": 0.5678, "conv_val_mean": 123.6,
            "n_not_converged_val": 0, "wall_time_mean": 5.0}
    base.update(kw)
    return pd.Series(base)


def test_fmt_conv_casos():
    assert F._fmt_conv(_summary_row(n_not_converged_val=5), 5) == "n/c (0/5)"
    assert F._fmt_conv(_summary_row(n_not_converged_val=0), 5) == "124 (5/5)"
    assert F._fmt_conv(_summary_row(n_not_converged_val=np.nan), 5) == "n/c"
    # convergidos pero sin media: n/c con total
    assert F._fmt_conv(_summary_row(n_not_converged_val=1, conv_val_mean=np.nan), 5) == "n/c (0/5)"


def test_build_table_redondeos():
    df = pd.DataFrame([_summary_row(), _summary_row(
        method="buy_and_hold", conv_val_mean=np.nan, n_not_converged_val=np.nan,
        wall_time_mean=np.nan, log_return_std_runs=np.nan)])
    t = F.build_table(df, n_runs=5)
    q, bh = t.iloc[0], t.iloc[1]
    assert q["Balance final (mediana)"] == "1235"
    assert q["% en mercado"] == "56.8"
    assert q["Drawdown máx. (%)"] == "30.0"
    assert q["Conv. val (episodio)"] == "124 (5/5)"
    assert q["Log-ret. medio ± std runs"] == "0.10 ± 0.02"
    assert bh["Conv. val (episodio)"] == "n/c"
    assert bh["Log-ret. medio ± std runs"] == "0.10"
    assert bh["Tiempo entren. (s)"] == "n/c"


def test_fig_politica_texto_celdas():
    # 5 runs: (pos0,bin0)=0 sin texto, (pos0,bin1)=1, (pos1,bin0)=0.6, (pos1,bin1)=0
    vals = {(0, 0): [0] * 5, (0, 1): [1] * 5, (1, 0): [1, 1, 1, 0, 0], (1, 1): [0] * 5}
    rows = [{"run": r, "bin": b, "pos": p, "target_position": v[r]}
            for (p, b), v in vals.items() for r in range(5)]
    df = pd.DataFrame(rows)
    policy = {a: {g: df for g in F.GENERATORS_ORDER} for a in F.ALGO_ORDER}
    fig = F.fig_politica(policy)
    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.texts) == [".6", "1"]
    plt.close(fig)


def test_plain_log_axis_menos_de_una_decada():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [100, 150])
    ax.set_yscale("log")
    F._plain_log_axis(ax)
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]
    assert len(ticks) >= 2
    fmt = ax.yaxis.get_major_formatter()
    assert all(fmt(t) != "" for t in ticks)
    plt.close(fig)


def test_fig_episodio_zorder_ql_sobre_sarsa(results):
    data = {g: F.load_episode_data(results, g, n_steps=60) for g in F.GENERATORS_ORDER}
    fig = F.fig_episodio(data)
    bals = [a for a in fig.axes if a.get_yscale() == "log"]
    assert bals
    for ax in bals:
        z = {c: max(ln.get_zorder() for ln in ax.lines if ln.get_color() == c)
             for c in (F.METHOD_COLORS["qlearning"], F.METHOD_COLORS["sarsa"])}
        assert z[F.METHOD_COLORS["qlearning"]] > z[F.METHOD_COLORS["sarsa"]]
    plt.close(fig)


def test_load_learning_data_sin_npz(results):
    for f in (results / F.GENERATORS_ORDER[0]).glob("train_sarsa_run*.npz"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match="sarsa"):
        F.load_learning_data(results)
