"""Tests de generadores y reparto de semillas."""
import numpy as np
import pytest

from rl_trader.prices import (FIT_SEED_BASE, GENERATORS, TEST_SEED_BASE,
                              TRAIN_RUN_STRIDE, VAL_SEED_BASE, make_series,
                              split_seeds, train_seed)

GENS = list(GENERATORS)


def test_cuatro_generadores():
    assert len(GENS) == 4


@pytest.mark.parametrize("gen", GENS)
def test_misma_semilla_identica(gen):
    a = make_series(gen, 7, 100)
    b = make_series(gen, 7, 100)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (100,)


@pytest.mark.parametrize("gen", GENS)
def test_distinta_semilla_distinta(gen):
    assert not np.array_equal(make_series(gen, 7, 100), make_series(gen, 8, 100))


@pytest.mark.parametrize("gen", GENS)
def test_make_series_independiente_del_estado_global(gen):
    np.random.seed(123)
    np.random.rand(10)
    a = make_series(gen, 3, 50)
    np.random.seed(999)
    b = make_series(gen, 3, 50)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("gen", GENS)
def test_precios_positivos(gen):
    assert (make_series(gen, 0, 500) > 0).all()


def test_rangos_split_disjuntos():
    n = 1000
    val = set(split_seeds("val", n))
    test = set(split_seeds("test", n))
    fit = set(split_seeds("fit", n))
    train = {train_seed(r, e) for r in range(10) for e in (0, 1, TRAIN_RUN_STRIDE - 1)}
    sets = [val, test, fit, train]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])
    assert max(train) < VAL_SEED_BASE <= min(val) and max(val) < TEST_SEED_BASE
    assert max(test) < FIT_SEED_BASE <= min(fit)


def test_split_seeds_consecutivas():
    assert split_seeds("test", 5) == [TEST_SEED_BASE + i for i in range(5)]
    assert split_seeds("val", 0) == []


def test_split_seeds_desconocido():
    with pytest.raises(KeyError):
        split_seeds("otro", 3)


def test_train_seed_valores():
    assert train_seed(0, 0) == 0
    assert train_seed(2, 5) == 2 * TRAIN_RUN_STRIDE + 5
    assert train_seed(9, TRAIN_RUN_STRIDE - 1) < VAL_SEED_BASE


def test_train_seed_asserts():
    with pytest.raises(AssertionError):
        train_seed(10, 0)
    with pytest.raises(AssertionError):
        train_seed(0, TRAIN_RUN_STRIDE)
