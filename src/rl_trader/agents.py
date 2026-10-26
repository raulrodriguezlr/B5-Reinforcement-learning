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
