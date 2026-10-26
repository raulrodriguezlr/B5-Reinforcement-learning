"""Generadores de precios del enunciado y reparto de semillas train/val/test."""
from typing import Callable

import numpy as np


def generar_precios(n_steps=1000):
    # Combinar una tendencia suave (drift) + componente cíclico + ruido
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)

    # Precio base acumulado
    retornos = drift + ciclo + ruido
    precios = 100 * np.cumprod(1 + retornos)
    return precios


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    """
    Genera precios usando Movimiento Browniano Geométrico.
    - mu: Tendencia o deriva esperada (drift).
    - sigma: Volatilidad (desviación estándar de los retornos).
    """
    dt = 1.0
    # Retornos logarítmicos distribuidos normalmente
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    precios = start_price * np.cumprod(np.exp(retornos))
    return precios


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    """
    Genera precios que tienden a revertir hacia un valor medio.
    - mean_price: El precio objetivo o de equilibrio.
    - kappa: Velocidad de reversión (cuánto más alto, más rápido vuelve al valor medio).
    - volatility: Magnitud del ruido aleatorio añadido en cada paso.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    for t in range(1, n_steps):
        # Fuerza de atracción hacia la media + ruido aleatorio
        delta = kappa * (mean_price - precios[t-1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta

        # Evitar precios negativos o cero por seguridad
        precios[t] = max(precios[t], 0.1)

    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    """
    Genera precios alternando entre regímenes alcistas y bajistas de forma estocástica.
    """
    precios = np.zeros(n_steps)
    precios[0] = start_price

    # Estado inicial: 0 = Alcista, 1 = Bajista
    estado = 0

    retornos = []
    for _ in range(n_steps):
        # Probabilidad de cambiar de régimen en cada paso
        if np.random.rand() < 0.02:
            estado = 1 - estado

        if estado == 0:  # Régimen alcista
            ret = np.random.normal(0.001, 0.01)
        else:            # Régimen bajista (caídas con más ruido)
            ret = np.random.normal(-0.0015, 0.025)

        retornos.append(ret)

    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)

    # Asegurar que no haya valores negativos
    precios = np.maximum(precios, 0.1)
    return precios


GENERATORS: dict[str, Callable] = {
    "tendencia_ciclo": generar_precios,
    "browniano": generar_precios_browniano,
    "reversion_media": generar_precios_reversion_media,
    "regimenes": generar_precios_regimenes,
}

# Rangos de semillas disjuntos (train < 10M <= val < 20M <= test < 30M <= fit)
TRAIN_RUN_STRIDE = 1_000_000
VAL_SEED_BASE = 10_000_000
TEST_SEED_BASE = 20_000_000
FIT_SEED_BASE = 30_000_000  # series para ajustar cortes; cuentan como entrenamiento

_SPLIT_BASES = {"val": VAL_SEED_BASE, "test": TEST_SEED_BASE, "fit": FIT_SEED_BASE}


def make_series(generator: str, seed: int, n_steps: int = 500) -> np.ndarray:
    """Genera una serie reproducible con el generador indicado."""
    # Los generadores usan np.random global: se siembra antes de cada serie
    np.random.seed(seed)
    return GENERATORS[generator](n_steps)


def train_seed(run_seed: int, episode: int) -> int:
    """Semilla de la serie del episodio `episode` en el run `run_seed`."""
    assert episode < TRAIN_RUN_STRIDE, "episodio fuera de rango"
    assert run_seed < 10, "run_seed debe ser < 10 para no invadir val"
    return run_seed * TRAIN_RUN_STRIDE + episode


def split_seeds(split: str, n: int) -> list[int]:
    """Lista de n semillas consecutivas para 'val', 'test' o 'fit'."""
    base = _SPLIT_BASES[split]
    return [base + i for i in range(n)]
