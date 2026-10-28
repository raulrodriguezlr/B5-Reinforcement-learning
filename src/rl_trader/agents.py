"""Agentes tabulares Q-Learning y Sarsa con política epsilon-greedy."""
import numpy as np


def epsilon_greedy(Q: np.ndarray, state: tuple[int, int], epsilon: float,
                   rng: np.random.Generator) -> int:
    """Acción aleatoria con prob. epsilon; si no, argmax con desempate aleatorio."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[-1]))
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


class TabularAgent:
    """Base con tabla Q (n_bins, 2, 3) y epsilon decreciente."""

    def __init__(self, n_bins: int, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.998, alpha_decay: float = 0.998,
                 alpha_min: float = 0.01, seed: int = 0):
        self.Q = np.zeros((n_bins, 2, 3))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.rng = np.random.default_rng(seed)

    def act(self, state: tuple[int, int]) -> int:
        return epsilon_greedy(self.Q, state, self.epsilon, self.rng)

    def greedy_action(self, state: tuple[int, int]) -> int:
        """Argmax determinista (primer máximo), para evaluar."""
        # Empate al primer máximo; estados no visitados (Q=0) eligen la acción 0
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_alpha(self) -> None:
        # Robbins-Monro: alpha decreciente estabiliza Q y la política greedy
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError


class QLearning(TabularAgent):
    def update(self, s: tuple[int, int], a: int, r: float, s2: tuple[int, int],
               done: bool) -> None:
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])


class Sarsa(TabularAgent):
    def update(self, s: tuple[int, int], a: int, r: float, s2: tuple[int, int],
               a2: int, done: bool) -> None:
        target = r if done else r + self.gamma * self.Q[s2][a2]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])


class MonteCarlo(TabularAgent):
    """Monte Carlo first-visit on-policy con política ε-greedy y media muestral."""
    episodic = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = np.zeros_like(self.Q)

    def update_episode(self, traj: list[tuple[tuple[int, int], int, float]]) -> None:
        """traj: lista de (estado, acción, recompensa) del episodio."""
        returns, G = [], 0.0
        for _, _, r in reversed(traj):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        seen = set()
        for (s, a, _), g in zip(traj, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            self.N[s][a] += 1
            self.Q[s][a] += (g - self.Q[s][a]) / self.N[s][a]


class MonteCarloOff(TabularAgent):
    """Monte Carlo off-policy con muestreo por importancia ponderado (comportamiento ε-greedy, objetivo greedy)."""
    episodic = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = np.zeros_like(self.Q)

    def update_episode(self, traj: list[tuple[tuple[int, int], int, float]]) -> None:
        """Recorre el episodio hacia atrás y corta en la primera acción que el objetivo no habría tomado."""
        behaviour_Q = self.Q.copy()  # Q con la que se generó el episodio
        G, W = 0.0, 1.0
        for s, a, r in reversed(traj):
            G = r + self.gamma * G
            self.C[s][a] += W
            self.Q[s][a] += W / self.C[s][a] * (G - self.Q[s][a])
            if a != int(np.argmax(self.Q[s])):
                break
            q = behaviour_Q[s]
            best = np.flatnonzero(q == q.max())
            p = self.epsilon / len(q) + ((1 - self.epsilon) / len(best) if a in best else 0.0)
            W /= p


def target_position(Q: np.ndarray) -> np.ndarray:
    """Posición resultante de la acción greedy, forma (n_bins, 2), int 0/1."""
    # Vender/mantener sin activo y mantener/comprar con activo son equivalentes
    # en efecto: solo cuenta la posición final, no la etiqueta de la acción
    greedy = np.argmax(Q, axis=-1)
    target = np.zeros(greedy.shape, dtype=np.int64)
    target[:, 0] = (greedy[:, 0] == 2).astype(np.int64)
    target[:, 1] = (greedy[:, 1] != 0).astype(np.int64)
    return target


ALGORITHMS = {"qlearning": QLearning, "sarsa": Sarsa}
# Extensiones: no entran en los experimentos por defecto
EXTRA_ALGORITHMS = {"mc": MonteCarlo, "mc_off": MonteCarloOff}
ALL_ALGORITHMS = {**ALGORITHMS, **EXTRA_ALGORITHMS}
