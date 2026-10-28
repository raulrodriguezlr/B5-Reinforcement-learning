"""Bucle de entrenamiento y criterios de convergencia."""
import time
from dataclasses import dataclass

import numpy as np

from rl_trader.agents import ALL_ALGORITHMS, Sarsa, TabularAgent, target_position
from rl_trader.discretize import TrendDiscretizer
from rl_trader.env_variants import BASE_COST_RATE, make_env
from rl_trader.evaluate import INITIAL_BALANCE, run_episode
from rl_trader.prices import make_series, train_seed


@dataclass
class TrainResult:
    episode_rewards: np.ndarray
    episode_final_balances: np.ndarray
    epsilons: np.ndarray
    alphas: np.ndarray
    policy_snapshots: np.ndarray  # (n_episodes, n_bins, 2) uint8
    cumulative_time: np.ndarray   # s acumulados al final de cada episodio
    wall_time: float
    Q: np.ndarray
    convergence_episode_policy: int | None
    convergence_episode_ma: int | None
    convergence_time_policy: float | None  # s
    val_episodes: np.ndarray            # índices (0-based) de episodios evaluados en val
    val_median_log_return: np.ndarray   # mediana de ln(final/1000) en val por punto
    val_time: float                     # s totales de evaluación en val (fuera de cumulative_time)
    convergence_episode_val: int | None
    convergence_time_val: float | None  # cumulative_time en convergence_episode_val
    convergence_time_ma: float | None = None  # cumulative_time en convergence_episode_ma


def policy_convergence_episode(snapshots: np.ndarray, window: int = 200,
                               agreement: float = 0.95) -> int | None:
    """Primer episodio desde el que la política coincide en >= agreement de celdas con la final; None si el tramo < window."""
    n = len(snapshots)
    agree = (snapshots == snapshots[-1]).reshape(n, -1).mean(axis=1)
    bad = np.flatnonzero(agree < agreement)
    start = int(bad[-1]) + 1 if len(bad) else 0
    return start if n - start >= window else None


def ma_convergence_episode(rewards: np.ndarray, window: int = 100,
                           tol_frac: float = 0.3) -> int | None:
    """Primer episodio (fin de ventana) desde el que la media móvil queda a <= tol_frac*std de la final.

    Mide la recompensa de entrenamiento (con ε), y la tolerancia usa la std de todo el entrenamiento.
    Con window=100, tol_frac=0.3 son 3 errores estándar de la media móvil (0.1 es 1 y el ruido solo ya lo supera).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if len(rewards) < window:
        return None
    csum = np.concatenate(([0.0], np.cumsum(rewards)))
    ma = (csum[window:] - csum[:-window]) / window
    tol = tol_frac * rewards.std()
    bad = np.flatnonzero(np.abs(ma - ma[-1]) > tol)
    first = int(bad[-1]) + 1 if len(bad) else 0
    return first + window - 1


def val_convergence_episode(val_episodes: np.ndarray, values: np.ndarray, n_episodes: int,
                            tol_abs: float = 0.01, tol_rel: float = 0.05,
                            window: int = 200) -> int | None:
    """Primer episodio evaluado desde el que el rendimiento en val queda a <= max(tol_abs, tol_rel*|final|) del final.

    None si el tramo estable (hasta el último episodio) dura menos de window episodios.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    tol = max(tol_abs, tol_rel * abs(values[-1]))
    bad = np.flatnonzero(np.abs(values - values[-1]) > tol)
    i = int(bad[-1]) + 1 if len(bad) else 0
    ep = int(val_episodes[i])
    return ep if n_episodes - 1 - ep >= window else None


def _val_median_log_return(agent: TabularAgent, discretizer: TrendDiscretizer,
                           val_series: list[np.ndarray],
                           cost_rate: float = BASE_COST_RATE) -> float:
    """Mediana de ln(final/1000) de la política greedy en las series de val (sin exploración ni updates)."""
    def policy(obs, env):
        return agent.greedy_action(discretizer.state(obs))
    logs = [np.log(run_episode(p, policy, cost_rate).balances[-1] / INITIAL_BALANCE) for p in val_series]
    return float(np.median(logs))


def train(algo: str, generator: str | list[str], discretizer: TrendDiscretizer, n_episodes: int,
          run_seed: int, n_steps: int = 500, alpha: float = 0.1, gamma: float = 0.95,
          epsilon_start: float = 1.0, epsilon_min: float = 0.05,
          epsilon_decay: float = 0.998, alpha_decay: float = 0.998,
          alpha_min: float = 0.01, policy_window: int = 200,
          policy_agreement: float = 0.95, ma_window: int = 100,
          ma_tol_frac: float = 0.3, val_series: list[np.ndarray] | None = None,
          val_every: int = 50, val_tol_abs: float = 0.01, val_tol_rel: float = 0.05,
          val_window: int = 200, cost_rate: float = BASE_COST_RATE) -> TrainResult:
    """Entrena un agente tabular con una serie nueva por episodio; con val_series evalúa la política greedy cada val_every episodios.

    Si generator es una lista, los episodios alternan los generadores en orden.
    """
    if val_series and val_every < 1:
        raise ValueError(f"val_every debe ser >= 1 (recibido {val_every})")
    n_bins = discretizer.n_bins
    agent = ALL_ALGORITHMS[algo](n_bins, alpha=alpha, gamma=gamma,
                             epsilon_start=epsilon_start, epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay, alpha_decay=alpha_decay,
                             alpha_min=alpha_min, seed=run_seed)
    is_sarsa = isinstance(agent, Sarsa)
    episodic = getattr(agent, "episodic", False)
    gens = [generator] if isinstance(generator, str) else list(generator)

    rewards = np.zeros(n_episodes)
    balances = np.zeros(n_episodes)
    epsilons = np.zeros(n_episodes)
    alphas = np.zeros(n_episodes)
    snapshots = np.zeros((n_episodes, n_bins, 2), dtype=np.uint8)
    cum_time = np.zeros(n_episodes)
    val_eps: list[int] = []
    val_vals: list[float] = []
    val_time = 0.0  # evaluación en val, excluida de cum_time

    t0 = time.perf_counter()
    for e in range(n_episodes):
        prices = make_series(gens[e % len(gens)], train_seed(run_seed, e), n_steps)
        env = make_env(prices, cost_rate)
        obs, _ = env.reset()
        s = discretizer.state(obs)
        a = agent.act(s)
        ep_reward = 0.0
        done = False
        traj: list = []
        while not done:
            obs, r, done, _, info = env.step(a)
            s2 = discretizer.state(obs)
            ep_reward += r
            if episodic:
                traj.append((s, a, r))
                a2 = agent.act(s2)
            elif is_sarsa:
                # a2 se elige antes de actualizar y es la acción siguiente
                a2 = agent.act(s2)
                agent.update(s, a, r, s2, a2, done)
            else:
                agent.update(s, a, r, s2, done)
                a2 = agent.act(s2)
            s, a = s2, a2
        if episodic:
            agent.update_episode(traj)
        rewards[e] = ep_reward
        balances[e] = info["balance"]
        epsilons[e] = agent.epsilon
        alphas[e] = agent.alpha
        agent.decay_epsilon()
        agent.decay_alpha()
        snapshots[e] = target_position(agent.Q)
        if val_series and ((e + 1) % val_every == 0 or e == n_episodes - 1):
            tv = time.perf_counter()
            val_eps.append(e)
            val_vals.append(_val_median_log_return(agent, discretizer, val_series, cost_rate))
            val_time += time.perf_counter() - tv
        cum_time[e] = time.perf_counter() - t0 - val_time
    wall = time.perf_counter() - t0 - val_time

    conv_pol = policy_convergence_episode(snapshots, policy_window, policy_agreement)
    conv_ma = ma_convergence_episode(rewards, ma_window, ma_tol_frac)
    conv_time = float(cum_time[conv_pol]) if conv_pol is not None else None
    val_episodes = np.array(val_eps, dtype=int)
    val_median = np.array(val_vals, dtype=np.float64)
    conv_val = (val_convergence_episode(val_episodes, val_median, n_episodes,
                                        val_tol_abs, val_tol_rel, val_window)
                if val_series else None)
    conv_time_val = float(cum_time[conv_val]) if conv_val is not None else None
    conv_time_ma = float(cum_time[conv_ma]) if conv_ma is not None else None
    return TrainResult(rewards, balances, epsilons, alphas, snapshots, cum_time, wall,
                       agent.Q.copy(), conv_pol, conv_ma, conv_time,
                       val_episodes, val_median, val_time, conv_val, conv_time_val, conv_time_ma)
