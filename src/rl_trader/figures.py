"""Figuras y tabla resumen para el informe (funciones puras sobre arrays/DataFrames)."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

from rl_trader.agents import ALGORITHMS
from rl_trader.baselines import buy_and_hold_factory, random_policy_factory
from rl_trader.discretize import TrendDiscretizer
from rl_trader.evaluate import run_episode
from rl_trader.experiments import add_ma_convergence
from rl_trader.prices import make_series, split_seeds

GENERATORS_ORDER = ["tendencia_ciclo", "browniano", "reversion_media", "regimenes"]
GENERATOR_LABELS = {
    "tendencia_ciclo": "Tendencia + ciclo",
    "browniano": "Browniano",
    "reversion_media": "Reversión a la media",
    "regimenes": "Regímenes",
}
METHOD_LABELS = {
    "qlearning": "Q-learning",
    "sarsa": "Sarsa",
    "buy_and_hold": "Buy & hold",
    "random": "Aleatorio",
}
METHOD_COLORS = {
    "qlearning": "#2a78d6",
    "sarsa": "#eb6834",
    "buy_and_hold": "#1baf7a",
    "random": "#898781",
}
METHOD_ORDER = ["qlearning", "sarsa", "buy_and_hold", "random"]
ALGO_ORDER = ["qlearning", "sarsa"]

BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"
LW = 1.5
WIDTH = 6.7  # in (17 cm)

_RC = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "lines.linewidth": LW, "axes.grid": False,
}


def _style(ax: Axes, grid: bool = True) -> None:
    """Estilo común: sin spines top/right, grid hairline solo en y."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, length=3)
    if grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)


ZORDER = {"qlearning": 4, "sarsa": 3, "buy_and_hold": 2, "random": 1}


def _method_line(method: str) -> dict:
    # Sarsa más gruesa por debajo de Q-learning para que ambas se vean si coinciden
    lw = LW * 2 if method == "sarsa" else LW
    return {"color": METHOD_COLORS[method], "linewidth": lw, "zorder": ZORDER[method],
            "linestyle": "--" if method == "random" else "-"}


def _method_handles(methods: list[str]) -> list[Line2D]:
    return [Line2D([], [], label=METHOD_LABELS[m], **_method_line(m)) for m in methods]


def _top_legend(fig: Figure, handles: list, ncol: int) -> None:
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False,
               labelcolor=INK)


# ---------------------------------------------------------------- carga de datos

def load_episode_data(results_dir: Path, generator: str, n_steps: int = 500,
                      random_seed: int = 0) -> dict:
    """Primera serie de test: posiciones greedy (run 0) y balances de los cuatro métodos."""
    gdir = Path(results_dir) / generator
    with open(gdir / "discretizer.json", encoding="utf-8") as f:
        disc = TrendDiscretizer.from_dict(json.load(f))
    seed = split_seeds("test", 1)[0]
    prices = make_series(generator, seed, n_steps)
    out: dict = {"prices": prices, "balances": {}}
    for algo in ALGO_ORDER:
        agent = ALGORITHMS[algo](disc.n_bins)
        agent.Q = np.load(gdir / f"train_{algo}_run0.npz")["Q"]
        res = run_episode(prices, lambda obs, env, a=agent: a.greedy_action(disc.state(obs)))
        out["balances"][algo] = res.balances
        if algo == "qlearning":
            out["positions"] = res.positions
    out["balances"]["buy_and_hold"] = run_episode(prices, buy_and_hold_factory(seed)).balances
    out["balances"]["random"] = run_episode(
        prices, random_policy_factory(random_seed)(seed)).balances
    return out


def load_policy_data(results_dir: Path) -> dict:
    """policy[algo][generador] = DataFrame de policy_<algo>.csv."""
    return {a: {g: pd.read_csv(Path(results_dir) / g / f"policy_{a}.csv")
                for g in GENERATORS_ORDER} for a in ALGO_ORDER}


def load_learning_data(results_dir: Path) -> dict:
    """learning[generador][algo] = {'episodes': (n,), 'values': (runs, n)}."""
    out: dict = {}
    for g in GENERATORS_ORDER:
        out[g] = {}
        for a in ALGO_ORDER:
            files = sorted((Path(results_dir) / g).glob(f"train_{a}_run*.npz"))
            if not files:
                raise FileNotFoundError(f"sin train_{a}_run*.npz en {Path(results_dir) / g}")
            zs = [np.load(f) for f in files]
            out[g][a] = {"episodes": zs[0]["val_episodes"],
                         "values": np.stack([z["val_median_log_return"] for z in zs])}
    return out


def load_test_data(results_dir: Path) -> dict:
    """test[generador][método] = log_return de todas las filas (run x serie)."""
    return {g: {m: pd.read_csv(Path(results_dir) / g / f"metrics_{m}.csv")["log_return"].to_numpy()
                for m in METHOD_ORDER} for g in GENERATORS_ORDER}


# ---------------------------------------------------------------- figuras

def _shade_runs(ax: Axes, positions: np.ndarray, color: str, alpha: float = 0.15) -> None:
    """Sombrea tramos con posición 1; positions[j] cubre el intervalo de precio [5+j, 6+j]."""
    pos = np.asarray(positions).astype(int)
    d = np.diff(np.concatenate(([0], pos, [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        ax.axvspan(5 + s, 5 + e, color=color, alpha=alpha, linewidth=0)


def _plain_log_axis(ax: Axes) -> None:
    """Eje y log con etiquetas planas (sin notación científica), al menos dos visibles."""
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(plain)
    ax.figure.canvas.draw()
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]
    if len(ticks) < 2:
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ticks = [t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]
    if len(ticks) < 2:
        # Rango menor que una década: 3 marcas repartidas en log y redondeadas a 2 cifras
        vals = np.geomspace(lo, hi, 5)[1:-1]
        ax.yaxis.set_major_locator(FixedLocator([float(f"{v:.2g}") for v in vals]))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(plain)


def fig_episodio(data: dict) -> Figure:
    """2x4: precio con posición de Q-learning y balances (escala log) en la serie de test."""
    gens = [g for g in GENERATORS_ORDER if g in data]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, len(gens), figsize=(WIDTH, 4.2), sharex="col",
                                 squeeze=False, layout="constrained")
        for j, g in enumerate(gens):
            d = data[g]
            top, bot = axes[0, j], axes[1, j]
            top.plot(np.arange(len(d["prices"])), d["prices"], color=INK2, linewidth=LW)
            _shade_runs(top, d["positions"], METHOD_COLORS["qlearning"])
            for m in METHOD_ORDER:
                b = d["balances"][m]
                bot.plot(5 + np.arange(len(b)), b, **_method_line(m))
            bot.set_yscale("log")
            _plain_log_axis(bot)
            top.set_title(GENERATOR_LABELS[g])
            bot.set_xlabel("paso")
            _style(top)
            _style(bot)
            if j == 0:
                top.set_ylabel("precio")
                bot.set_ylabel("balance (escala log)")
        handles = _method_handles(METHOD_ORDER) + [
            Patch(facecolor=METHOD_COLORS["qlearning"], alpha=0.15,
                  label="Q-learning en el mercado")]
        _top_legend(fig, handles, ncol=5)
    return fig


def _policy_matrix(df: pd.DataFrame, n_bins: int) -> np.ndarray:
    """Media de target_position entre runs, forma (2, n_bins): filas pos 0 y 1."""
    mat = df.groupby(["pos", "bin"])["target_position"].mean().unstack("bin")
    return mat.reindex(index=[0, 1], columns=range(n_bins)).to_numpy(dtype=float)


def fig_politica(policy: dict) -> Figure:
    """Heatmaps 2 (algoritmo) x 4 (generador) de la fracción de runs que entra en el mercado."""
    gens = [g for g in GENERATORS_ORDER if g in policy[ALGO_ORDER[0]]]
    cmap = LinearSegmentedColormap.from_list("azul", [BG, "#104281"])
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, len(gens), figsize=(WIDTH, 3.4), squeeze=False,
                                 layout="constrained")
        mesh = None
        for i, a in enumerate(ALGO_ORDER):
            for j, g in enumerate(gens):
                ax = axes[i, j]
                df = policy[a][g]
                n_bins = int(df["bin"].max()) + 1
                mat = _policy_matrix(df, n_bins)
                mesh = ax.pcolormesh(mat, cmap=cmap, vmin=0, vmax=1,
                                     edgecolors=BG, linewidth=1.5)
                # Celdas con valor 0 en gris claro para distinguirlas del fondo
                zero = np.ma.masked_where(mat != 0, np.zeros_like(mat))
                ax.pcolormesh(zero, cmap=LinearSegmentedColormap.from_list("g", [GRID, GRID]),
                              edgecolors=BG, linewidth=1.5)
                ax.set_xlim(0, n_bins)
                ax.set_ylim(2, 0)
                for r in range(2):
                    for c in range(n_bins):
                        v = mat[r, c]
                        if np.isnan(v) or v == 0:
                            continue
                        txt = "1" if v == 1 else f"{v:.1f}".lstrip("0")
                        ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center",
                                fontsize=5.5, color="white" if v > 0.6 else INK)
                _style(ax, grid=False)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_xticks(np.arange(n_bins) + 0.5, [str(k) for k in range(n_bins)])
                ax.tick_params(length=0, labelsize=6)
                if j == 0:
                    ax.set_yticks([0.5, 1.5], ["sin activo", "con activo"])
                    ax.set_ylabel(METHOD_LABELS[a], fontsize=8, color=INK)
                else:
                    ax.set_yticks([])
                if i == 0:
                    ax.set_title(GENERATOR_LABELS[g])
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("tramo de tendencia", labelpad=17)
                    ax.text(0, -0.2, "baja", transform=ax.transAxes, ha="left", va="top",
                            fontsize=6, color=INK2)
                    ax.text(1, -0.2, "alta", transform=ax.transAxes, ha="right", va="top",
                            fontsize=6, color=INK2)
        cb = fig.colorbar(mesh, ax=axes, location="bottom", shrink=0.5, aspect=40, pad=0.02)
        cb.set_label("fracción de runs en el mercado", color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=6.5, length=2)
    return fig


def fig_aprendizaje(learning: dict) -> Figure:
    """1x4: mediana de log-retorno en validación (media de runs y banda min-max)."""
    gens = [g for g in GENERATORS_ORDER if g in learning]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(gens), figsize=(WIDTH, 2.4), squeeze=False,
                                 layout="constrained")
        for j, g in enumerate(gens):
            ax = axes[0, j]
            for a in ALGO_ORDER:
                x = np.asarray(learning[g][a]["episodes"]) + 1
                v = np.asarray(learning[g][a]["values"])
                c = METHOD_COLORS[a]
                ax.fill_between(x, v.min(axis=0), v.max(axis=0), color=c, alpha=0.15,
                                linewidth=0)
                ax.plot(x, v.mean(axis=0), color=c, linewidth=LW)
            ax.set_title(GENERATOR_LABELS[g])
            ax.set_xlabel("episodio")
            if j == 0:
                ax.set_ylabel("log-retorno mediano (validación)")
            _style(ax)
        _top_legend(fig, _method_handles(ALGO_ORDER), ncol=2)
    return fig


def fig_test(test: dict) -> Figure:
    """1x4: boxplots del log-retorno en test por método."""
    gens = [g for g in GENERATORS_ORDER if g in test]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(gens), figsize=(WIDTH, 2.6), squeeze=False,
                                 layout="constrained")
        for j, g in enumerate(gens):
            ax = axes[0, j]
            ax.axhline(0, color=AXIS, linewidth=0.5, zorder=1)
            bp = ax.boxplot([test[g][m] for m in METHOD_ORDER], patch_artist=True,
                            widths=0.6, whis=1.5,
                            flierprops={"marker": ".", "markersize": 1.5,
                                        "markerfacecolor": AXIS, "markeredgecolor": AXIS,
                                        "alpha": 0.5})
            for k, m in enumerate(METHOD_ORDER):
                c = METHOD_COLORS[m]
                bp["boxes"][k].set(facecolor=c, alpha=0.3, edgecolor=c, linewidth=1.0)
                for w in bp["whiskers"][2 * k:2 * k + 2] + bp["caps"][2 * k:2 * k + 2]:
                    w.set(color=c, linewidth=1.0)
                bp["medians"][k].set(color=c, linewidth=LW)
            ax.set_xticks([])
            ax.set_title(GENERATOR_LABELS[g])
            if j == 0:
                ax.set_ylabel("log-retorno (test)")
            _style(ax)
        handles = [Patch(facecolor=METHOD_COLORS[m], alpha=0.3,
                         edgecolor=METHOD_COLORS[m], label=METHOD_LABELS[m])
                   for m in METHOD_ORDER]
        _top_legend(fig, handles, ncol=4)
    return fig


# ---------------------------------------------------------------- tabla

TABLE_COLUMNS = ["Generador", "Método", "Episodios / ε_min", "Balance final (mediana)",
                 "Log-ret. medio ± std runs", "Exceso vs B&H (pp, mediana)", "Sharpe",
                 "Drawdown máx. (%)", "Operaciones", "% en mercado",
                 "Conv. val (episodio)", "Conv. MA (episodio)", "Conv. MA (s)",
                 "Tiempo entren. (s)"]


def _fmt(v, spec: str) -> str:
    return "n/c" if v is None or pd.isna(v) else format(v, spec)


def _fmt_conv(r: pd.Series, n_runs: int, key: str = "val") -> str:
    """Convergencia ('val' o 'ma'): 'media (convergidos/total)' o 'n/c (0/total)'."""
    nnc = r.get(f"n_not_converged_{key}")
    if nnc is None or pd.isna(nnc):
        return "n/c"
    ok = n_runs - int(nnc)
    mean = r.get("conv_val_mean" if key == "val" else "conv_episode_ma_mean")
    if ok <= 0 or mean is None or pd.isna(mean):
        return f"n/c (0/{n_runs})"
    return f"{mean:.0f} ({ok}/{n_runs})"


def _fmt_budget(r: pd.Series) -> str:
    """'N / ε_min' de los agentes; n/c en baselines."""
    n, em = r.get("n_episodes"), r.get("epsilon_min")
    if n is None or em is None or pd.isna(n) or pd.isna(em):
        return "n/c"
    return f"{n:.0f} / {em:g}"


def build_table(summary: pd.DataFrame, n_runs: int = 5) -> pd.DataFrame:
    """Tabla resumen (strings) desde summary.csv; medianas salvo log-ret y tiempo (medias)."""
    rows = []
    for g in GENERATORS_ORDER:
        for m in METHOD_ORDER:
            sel = summary[(summary["generator"] == g) & (summary["method"] == m)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            lr = _fmt(r["log_return_mean"], ".2f")
            if not pd.isna(r["log_return_std_runs"]):
                lr += f" ± {r['log_return_std_runs']:.2f}"
            dd = r["max_drawdown_median"]
            tm = r["time_in_market_median"]
            rows.append({
                "Generador": GENERATOR_LABELS[g],
                "Método": METHOD_LABELS[m],
                "Episodios / ε_min": _fmt_budget(r),
                "Balance final (mediana)": _fmt(r["final_balance_median"], ".0f"),
                "Log-ret. medio ± std runs": lr,
                "Exceso vs B&H (pp, mediana)": _fmt(r["excess_return_pct_median"], ".1f"),
                "Sharpe": _fmt(r["sharpe_median"], ".3f"),
                "Drawdown máx. (%)": _fmt(None if pd.isna(dd) else dd * 100, ".1f"),
                "Operaciones": _fmt(r["n_trades_median"], ".0f"),
                "% en mercado": _fmt(None if pd.isna(tm) else tm * 100, ".1f"),
                "Conv. val (episodio)": _fmt_conv(r, n_runs),
                "Conv. MA (episodio)": _fmt_conv(r, n_runs, "ma"),
                "Conv. MA (s)": _fmt(r.get("conv_time_ma_mean"), ".1f"),
                "Tiempo entren. (s)": _fmt(r.get("wall_time_mean"), ".1f"),
            })
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _lr_cell(df: pd.DataFrame, generator: str, method: str) -> str:
    """Log-retorno medio ± std entre semillas de una fila de resumen; n/c si falta."""
    sel = df[(df["generator"] == generator) & (df["method"] == method)]
    if sel.empty:
        return "n/c"
    r = sel.iloc[0]
    out = _fmt(r["log_return_mean"], ".2f")
    if not pd.isna(r["log_return_std_runs"]):
        out += f" ± {r['log_return_std_runs']:.2f}"
    return out


def build_ext_tables(final: pd.DataFrame, ext: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tablas de extensiones (strings): algoritmos y entrenamiento, y sensibilidad a tramos y comisión.

    ext: resúmenes por clave 'bins5', 'bins20', 'cost0', 'cost0.005', 'mc' y 'mixed'.
    """
    algos = []
    for g in GENERATORS_ORDER:
        algos.append({
            "Generador": GENERATOR_LABELS[g],
            "Q-learning": _lr_cell(final, g, "qlearning"),
            "Sarsa": _lr_cell(final, g, "sarsa"),
            "MC on-policy": _lr_cell(ext["mc"], g, "mc"),
            "MC off-policy": _lr_cell(ext["mc"], g, "mc_off"),
            "Q-learning mixto": _lr_cell(ext["mixed"], g, "qlearning_mixto"),
            "Sarsa mixto": _lr_cell(ext["mixed"], g, "sarsa_mixto"),
            "Buy & hold": _lr_cell(final, g, "buy_and_hold"),
        })
    sens = []
    for g in GENERATORS_ORDER:
        for m in ("qlearning", "sarsa"):
            row = {"Generador": GENERATOR_LABELS[g], "Método": METHOD_LABELS[m]}
            for label, df in (("5", ext["bins5"]), ("10", final), ("20", ext["bins20"])):
                row[f"{label} tramos"] = _lr_cell(df, g, m)
            for label, df in (("0 %", ext["cost0"]), ("0,1 %", final), ("0,5 %", ext["cost0.005"])):
                sel = df[(df["generator"] == g) & (df["method"] == m)]
                n = "n/c" if sel.empty else _fmt(sel.iloc[0]["n_trades_median"], ".0f")
                row[f"comisión {label}"] = f"{_lr_cell(df, g, m)} ({n})"
            sens.append(row)
    return {"algoritmos": pd.DataFrame(algos), "sensibilidad": pd.DataFrame(sens)}


def table_to_markdown(table: pd.DataFrame) -> str:
    """Tabla en formato Markdown (sin dependencias externas)."""
    head = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([head, sep, *body]) + "\n"


def save_figure(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Guarda PNG a 300 dpi y PDF vectorial."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, facecolor=BG)
    fig.savefig(paths[1], facecolor=BG)
    return paths


def make_all(results_dir: Path, out_dir: Path, n_steps: int = 500, random_seed: int = 0,
             n_runs: int = 5, ma_window: int = 100,
             ma_tol_frac: float = 0.3) -> list[Path]:
    """Genera todas las figuras y la tabla desde results/final; devuelve los ficheros."""
    matplotlib.use("Agg")
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    figs = {
        "fig_episodio": fig_episodio(
            {g: load_episode_data(results_dir, g, n_steps, random_seed) for g in GENERATORS_ORDER}),
        "fig_politica": fig_politica(load_policy_data(results_dir)),
        "fig_aprendizaje": fig_aprendizaje(load_learning_data(results_dir)),
        "fig_test": fig_test(load_test_data(results_dir)),
    }
    written: list[Path] = []
    for name, fig in figs.items():
        written += save_figure(fig, out_dir, name)
        plt.close(fig)
    summary = add_ma_convergence(pd.read_csv(results_dir / "summary.csv"), results_dir,
                                 ma_window, ma_tol_frac)
    table = build_table(summary, n_runs)
    table.to_csv(out_dir / "tabla_resumen.csv", index=False)
    (out_dir / "tabla_resumen.md").write_text(table_to_markdown(table), encoding="utf-8")
    return written + [out_dir / "tabla_resumen.csv", out_dir / "tabla_resumen.md"]
