"""Unit tests for oddfisher."""

from pathlib import Path

import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from pytest import mark

from oddfisher.fisher import (
    dhyper,
    compute_dnhyper,
    compute_mnhyper,
)


test_dir = Path('tests/data/')


@mark.parametrize('a, b, c, d, is_log, expected', [
    (1, 2, 3, 4, True, np.array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738])),
    (1, 2, 3, 4, False, np.array([0.16666667, 0.50000000, 0.30000000, 0.03333333])),
])
def test_dhyper(a: int, b: int, c: int, d: int, is_log: bool, expected: list[float]) -> None:
    """Test dhyper against R dhyper function, where expected is output from R."""
    data = np.array([a, b, c, d]).reshape((2,2))
    mn = data.sum(axis=1)
    M = sum(mn)
    n = mn[1]  # Total diseased, TP + FN
    N = data.sum(axis=0)[0]  # Number called diseased, TP + FP
    
    lo = max(0, N - n)
    hi = min(N, n)
    support = np.arange(lo, hi)
    
    observed = dhyper(support, M, M - n, N, is_log=is_log)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


@mark.parametrize('a, b, c, d, odd_ratio, expected', [
    (1, 2, 3, 4, 10, np.array([0.00243309, 0.07299270, 0.43795620, 0.48661800])),
    (1, 2, 3, 4, 1, np.array([0.16666667, 0.50000000, 0.30000000, 0.03333333])),
])
def test_compute_dnhyper(a: int, b: int, c: int, d: int, odd_ratio: int | float, expected: list[float]) -> None:
    """Test dhyper against R dhyper function, where expected is output from R."""
    data = np.array([a, b, c, d]).reshape((2,2))
    mn = data.sum(axis=1)
    M = sum(mn)
    n = mn[1]  # Total diseased, TP + FN
    N = data.sum(axis=0)[0]  # Number called diseased, TP + FP
    
    lo = max(0, N - n)
    hi = min(N, n)
    support = np.arange(lo, hi)

    observed = compute_dnhyper(support, M, M - n, N, odd_ratio=odd_ratio)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


@mark.parametrize('a, b, c, d, odd_ratio, expected', [
    (1, 2, 3, 4, 10, 2.408759),
    (1, 2, 3, 4, 1, 1.2),
    (1, 2, 3, 4, 0, 0),
    (1, 2, 3, 4, np.inf, 3),
])
def test_compute_mnhyper(a: int, b: int, c: int, d: int, odd_ratio: int | float, expected: int | float) -> None:
    """Test dhyper against R dhyper function, where expected is output from R."""
    data = np.array([a, b, c, d]).reshape((2,2))
    mn = data.sum(axis=1)
    M = sum(mn)
    n = mn[1]  # Total diseased, TP + FN
    N = data.sum(axis=0)[0]  # Number called diseased, TP + FP
    
    lo = max(0, N - n)
    hi = min(N, n)
    support = np.arange(lo, hi)

    observed = compute_mnhyper(support, M, M - n, N, odd_ratio=odd_ratio)
    print(observed, expected, lo, M, n, N)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

