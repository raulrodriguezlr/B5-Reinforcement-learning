"""Tests del entorno: equivalencia con el notebook y mecánica de compra/venta."""
import ast
import inspect
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from rl_trader import env as env_module
from rl_trader.env import SimpleTradingEnv
from rl_trader.prices import make_series

NOTEBOOK = Path(os.environ.get("TALLER_NOTEBOOK",
                               Path.home() / "Downloads" / "taller_trader.ipynb"))


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def _dump(cls_node: ast.AST) -> str:
    return ast.dump(_strip_docstrings(cls_node), include_attributes=False)


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook del enunciado no disponible")
def test_env_identico_al_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb_cls = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "class SimpleTradingEnv" not in src:
            continue
        # se descartan magics de Jupyter
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        for node in ast.parse(src).body:
            if isinstance(node, ast.ClassDef) and node.name == "SimpleTradingEnv":
                nb_cls = node
    assert nb_cls is not None
    repo_cls = ast.parse(textwrap.dedent(inspect.getsource(SimpleTradingEnv))).body[0]
    assert _dump(nb_cls) == _dump(repo_cls)


def test_modulo_env_sin_logica_extra():
    # solo la clase del enunciado como definición propia
    classes = [n for n, o in vars(env_module).items()
               if inspect.isclass(o) and o.__module__ == env_module.__name__]
    assert classes == ["SimpleTradingEnv"]


def test_episodio_500_precios_son_494_pasos():
    prices = make_series("browniano", 1, 500)
    env = SimpleTradingEnv(prices)
    env.reset()
    n, done = 0, False
    while not done:
        _, _, done, trunc, _ = env.step(1)
        n += 1
        assert trunc is False
    assert n == 494


def test_obs_inicial_y_dtype():
    env = SimpleTradingEnv(np.array([100.0] * 6 + [110.0, 99.0, 99.0]))
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.tolist() == [0.0, 0.0]
    assert info == {}


def test_compra_mantiene_vende_a_mano():
    p = np.array([100.0] * 6 + [110.0, 99.0, 99.0])
    env = SimpleTradingEnv(p)
    env.reset()

    # compra a 100, comisión 1 sobre 1000, 9.99 acciones
    obs, r, term, _, info = env.step(2)
    assert r == pytest.approx(-0.001 + 0.10)
    assert info["balance"] == pytest.approx(9.99 * 110)
    assert obs[1] == 1 and not term
    assert env.cash == 0.0 and env.shares == pytest.approx(9.99)

    # mantiene: solo retorno del precio
    _, r, term, _, info = env.step(1)
    assert r == pytest.approx((99 - 110) / 110)
    assert info["balance"] == pytest.approx(9.99 * 99)
    assert not term

    # vende a 99 (precio previo), comisión 0.1 %
    obs, r, term, _, info = env.step(0)
    assert r == pytest.approx(-0.001)
    assert info["balance"] == pytest.approx(9.99 * 99 * (1 - 0.001))
    assert obs[1] == 0 and term


def test_acciones_sin_efecto():
    env = SimpleTradingEnv(np.array([100.0] * 7 + [110.0, 99.0, 99.0]))
    env.reset()
    # vender sin posición: sin coste ni recompensa
    _, r, _, _, info = env.step(0)
    assert r == 0.0 and info["balance"] == 1000.0
    # comprar con posición abierta no vuelve a cobrar comisión
    env.step(2)
    _, r, _, _, _ = env.step(2)
    assert r == pytest.approx((99 - 110) / 110)


def test_obs_tendencia_clip():
    p = np.array([1.0] * 5 + [1000.0, 1000.0, 1000.0])
    env = SimpleTradingEnv(p)
    obs, _ = env.reset()
    # (1000 - media) / media > 1: se recorta a 1
    assert obs[0] == 1.0
