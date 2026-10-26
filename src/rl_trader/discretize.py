"""Discretización de la tendencia por cuantiles ajustados con series de entrenamiento."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rl_trader.prices import make_series, split_seeds


def trend_series(prices: np.ndarray) -> np.ndarray:
    """Tendencia como en env._get_obs para t = 5 .. len(prices)-2 (estados de decisión)."""
    windows = sliding_window_view(np.asarray(prices, dtype=np.float64), 6)[:-1]
    mean = windows.mean(axis=1)
    trend = (windows[:, -1] - mean) / mean
    return np.clip(trend, -1.0, 1.0).astype(np.float32)


class TrendDiscretizer:
    """Bins por cuantiles de la tendencia; la posición pasa tal cual."""

    def __init__(self, edges: np.ndarray | None = None):
        self.edges = None if edges is None else np.asarray(edges, dtype=np.float64)

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    def fit(self, series: list[np.ndarray], n_bins: int = 10) -> "TrendDiscretizer":
        """Calcula los cortes interiores como cuantiles de todas las tendencias."""
        all_trends = np.concatenate(series)
        self.edges = np.quantile(all_trends, np.linspace(0, 1, n_bins + 1)[1:-1])
        return self

    def bin(self, trend: float) -> int:
        # side="right": bin = nº de cortes <= tendencia; extremos abiertos
        return int(np.searchsorted(self.edges, trend, side="right"))

    def state(self, obs: np.ndarray) -> tuple[int, int]:
        return self.bin(float(obs[0])), int(obs[1])

    def to_dict(self) -> dict:
        return {"n_bins": self.n_bins, "edges": [float(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "TrendDiscretizer":
        return cls(np.array(d["edges"], dtype=np.float64))


def fit_discretizer(generator: str, n_series: int = 200, n_steps: int = 500,
                    n_bins: int = 10) -> TrendDiscretizer:
    """Ajusta cortes con series 'fit' del generador (nunca test)."""
    series = [trend_series(make_series(generator, s, n_steps))
              for s in split_seeds("fit", n_series)]
    return TrendDiscretizer().fit(series, n_bins=n_bins)
