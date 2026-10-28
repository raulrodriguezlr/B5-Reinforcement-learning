"""Orquestación de experimentos: entrenamiento, evaluación en test y tablas resumen."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rl_trader.agents import ALL_ALGORITHMS, target_position
from rl_trader.baselines import buy_and_hold_factory, random_policy_factory
from rl_trader.discretize import TrendDiscretizer, fit_discretizer, trend_series
from rl_trader.env_variants import BASE_COST_RATE
from rl_trader.evaluate import evaluate_policy
from rl_trader.prices import GENERATORS, make_series, split_seeds
from rl_trader.train import ma_convergence_episode, train


def load_config(path: Path) -> dict:
    """Lee la configuración JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Media y std (ddof=1) ignorando None; NaN si no hay datos."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _metric_row(generator: str, method: str, df: pd.DataFrame) -> dict:
    row = {"generator": generator, "method": method}
    for col in df.columns:
        if col in ("seed", "run"):
            continue
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_median"] = float(df[col].median())
        row[f"{col}_std_series"] = float(df[col].std())
        # std entre medias por run; NaN en baselines (sin run)
        row[f"{col}_std_runs"] = (float(df.groupby("run")[col].mean().std(ddof=1))
                                  if "run" in df.columns else float("nan"))
    return row


def _policy_table(algo: str, run: int, Q: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    """Política por (bin, posición) con límites del bin."""
    target = target_position(Q)
    lows = np.concatenate(([-np.inf], edges))
    highs = np.concatenate((edges, [np.inf]))
    rows = []
    for b in range(Q.shape[0]):
        for pos in (0, 1):
            rows.append({"algo": algo, "run": run, "bin": b, "pos": pos,
                         "target_position": int(target[b, pos]),
                         "greedy_action": int(np.argmax(Q[b, pos])),
                         "bin_low": lows[b], "bin_high": highs[b]})
    return pd.DataFrame(rows)


def run_generator(generator: str, cfg: dict, out_dir: Path,
                  algos: list[str],
                  agent_overrides: dict[str, dict] | None = None) -> pd.DataFrame:
    """Ajusta discretizador, entrena y evalúa cada algoritmo y baselines; agent_overrides pisa cfg["agent"] por algoritmo."""
    gen_dir = Path(out_dir) / generator
    gen_dir.mkdir(parents=True, exist_ok=True)
    n_steps = cfg["n_steps"]
    n_test = cfg["n_test_series"]
    cost_rate = cfg.get("cost_rate", BASE_COST_RATE)

    disc = fit_discretizer(generator, cfg["n_fit_series"], n_steps, cfg["n_bins"])
    with open(gen_dir / "discretizer.json", "w", encoding="utf-8") as f:
        json.dump(disc.to_dict(), f, indent=2)

    if cfg["n_val_series"] < 1:
        raise ValueError(f"n_val_series debe ser >= 1 (recibido {cfg['n_val_series']})")
    # Series de val generadas una vez por generador (nunca test)
    val_series = [make_series(generator, sd, n_steps)
                  for sd in split_seeds("val", cfg["n_val_series"])]

    summary = []
    for algo in algos:
        agent_params = {**cfg["agent"], **(agent_overrides or {}).get(algo, {})}
        n_episodes = agent_params.pop("n_episodes", cfg["n_episodes"])  # barrido: N por algoritmo
        frames, policies = [], []
        conv_pol, conv_ma, conv_time, walls = [], [], [], []
        conv_val, conv_val_time, val_times = [], [], []
        conv_ma_time = []
        for run in range(cfg["n_runs"]):
            res = train(algo, generator, disc, n_episodes, run, n_steps,
                        val_series=val_series, cost_rate=cost_rate,
                        **agent_params, **cfg["convergence"])
            val_final = (f"{res.val_median_log_return[-1]:.4f}"
                         if len(res.val_median_log_return) > 0 else "n/a")
            print(f"[{generator}] {algo} run {run}: wall {res.wall_time:.1f}s, "
                  f"conv_policy={res.convergence_episode_policy}, "
                  f"conv_ma={res.convergence_episode_ma}, "
                  f"conv_val={res.convergence_episode_val}, "
                  f"val_final={val_final}, "
                  f"val_time={res.val_time:.1f}s", flush=True)
            np.savez_compressed(gen_dir / f"train_{algo}_run{run}.npz",
                                rewards=res.episode_rewards,
                                balances=res.episode_final_balances,
                                epsilons=res.epsilons, alphas=res.alphas,
                                snapshots=res.policy_snapshots,
                                cumulative_time=res.cumulative_time, Q=res.Q,
                                val_episodes=res.val_episodes,
                                val_median_log_return=res.val_median_log_return)
            # Agente solo para reutilizar greedy_action con la Q final
            agent = ALL_ALGORITHMS[algo](disc.n_bins)
            agent.Q = res.Q

            def factory(series_seed: int, agent=agent):
                return lambda obs, env: agent.greedy_action(disc.state(obs))

            df = evaluate_policy(factory, generator, "test", n_test, n_steps, cost_rate)
            df.insert(0, "run", run)
            frames.append(df)
            policies.append(_policy_table(algo, run, res.Q, disc.edges))
            conv_pol.append(res.convergence_episode_policy)
            conv_ma.append(res.convergence_episode_ma)
            conv_ma_time.append(res.convergence_time_ma)
            conv_time.append(res.convergence_time_policy)
            walls.append(res.wall_time)
            conv_val.append(res.convergence_episode_val)
            conv_val_time.append(res.convergence_time_val)
            val_times.append(res.val_time)

        metrics = pd.concat(frames, ignore_index=True)
        metrics.to_csv(gen_dir / f"metrics_{algo}.csv", index=False)
        pd.concat(policies, ignore_index=True).to_csv(gen_dir / f"policy_{algo}.csv", index=False)

        row = _metric_row(generator, algo, metrics)
        for k in ("alpha", "gamma", "epsilon_decay", "epsilon_min"):
            row[k] = agent_params[k]
        row["n_episodes"] = n_episodes
        row["conv_episode_policy_mean"], row["conv_episode_policy_std"] = _mean_std(conv_pol)
        row["conv_episode_ma_mean"], row["conv_episode_ma_std"] = _mean_std(conv_ma)
        row["conv_time_ma_mean"], row["conv_time_ma_std"] = _mean_std(conv_ma_time)
        row["conv_time_policy_mean"], row["conv_time_policy_std"] = _mean_std(conv_time)
        row["conv_val_mean"], row["conv_val_std"] = _mean_std(conv_val)
        row["conv_val_time_mean"], row["conv_val_time_std"] = _mean_std(conv_val_time)
        row["val_time_mean"], _ = _mean_std(val_times)
        row["wall_time_mean"], row["wall_time_std"] = _mean_std(walls)
        row["n_not_converged_policy"] = sum(v is None for v in conv_pol)
        row["n_not_converged_ma"] = sum(v is None for v in conv_ma)
        row["n_not_converged_val"] = sum(v is None for v in conv_val)
        summary.append(row)

    # Baselines sobre las mismas series de test
    baselines = {
        "random": random_policy_factory(cfg["random_base_seed"]),
        "buy_and_hold": buy_and_hold_factory,
    }
    for name, factory in baselines.items():
        df = evaluate_policy(factory, generator, "test", n_test, n_steps, cost_rate)
        df.to_csv(gen_dir / f"metrics_{name}.csv", index=False)
        summary.append(_metric_row(generator, name, df))

    return pd.DataFrame(summary)


def add_ma_convergence(summary: pd.DataFrame, results_dir: Path, window: int = 100,
                       tol_frac: float = 0.3) -> pd.DataFrame:
    """Recalcula episodio y tiempo de convergencia por media móvil desde los train_*.npz, sin reentrenar."""
    out = summary.copy()
    for col in ("conv_episode_ma_mean", "conv_episode_ma_std", "conv_time_ma_mean",
                "conv_time_ma_std", "n_not_converged_ma"):
        out[col] = np.nan
    for i, r in out.iterrows():
        if r["method"] not in ALL_ALGORITHMS:
            continue
        eps, times = [], []
        for f in sorted((Path(results_dir) / r["generator"]).glob(f"train_{r['method']}_run*.npz")):
            with np.load(f) as d:
                e = ma_convergence_episode(d["rewards"], window, tol_frac)
                eps.append(e)
                times.append(None if e is None else float(d["cumulative_time"][e]))
        out.loc[i, ["conv_episode_ma_mean", "conv_episode_ma_std"]] = _mean_std(eps)
        out.loc[i, ["conv_time_ma_mean", "conv_time_ma_std"]] = _mean_std(times)
        out.loc[i, "n_not_converged_ma"] = sum(e is None for e in eps)
    return out


def run_mixed(cfg: dict, out_dir: Path, algos: list[str], n_episodes: int,
              params: dict | None = None) -> pd.DataFrame:
    """Entrena un agente por algoritmo y semilla alternando los cuatro generadores y lo evalúa en test de cada uno."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps, gens = cfg["n_steps"], list(GENERATORS)
    fit = [trend_series(make_series(g, sd, n_steps))
           for g in gens for sd in split_seeds("fit", cfg["n_fit_series"])]
    disc = TrendDiscretizer().fit(fit, n_bins=cfg["n_bins"])
    agent_params = {**cfg["agent"], **(params or {})}
    summary = []
    for algo in algos:
        frames = []
        for run in range(cfg["n_runs"]):
            res = train(algo, gens, disc, n_episodes, run, n_steps, **agent_params)
            print(f"[mixto] {algo} run {run}: wall {res.wall_time:.1f}s", flush=True)
            agent = ALL_ALGORITHMS[algo](disc.n_bins)
            agent.Q = res.Q

            def factory(series_seed: int, agent=agent):
                return lambda obs, env: agent.greedy_action(disc.state(obs))

            for g in gens:
                df = evaluate_policy(factory, g, "test", cfg["n_test_series"], n_steps)
                df.insert(0, "run", run)
                df.insert(0, "generator", g)
                frames.append(df)
        metrics = pd.concat(frames, ignore_index=True)
        metrics.to_csv(out_dir / f"metrics_mixed_{algo}.csv", index=False)
        for g in gens:
            summary.append(_metric_row(g, f"{algo}_mixto", metrics[metrics["generator"] == g]
                                       .drop(columns="generator")))
    return pd.DataFrame(summary)
