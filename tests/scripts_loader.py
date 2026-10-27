"""Carga scripts/make_figures.py como módulo para los tests."""
import importlib.util
from pathlib import Path
from types import ModuleType


def load_make_figures() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
