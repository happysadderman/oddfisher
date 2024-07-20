# oddfisher.py, a python version of R fisher exact test with od parameter.

import os
import sys
import argparse

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import hypergeom
from scipy.optimize import brentq


def check_input() -> None:
    """Check imput.
    
    Args:
        data: data in the format of 2x2 matrix
        
    Raises:
        ValueError when the data is not in 2x2 format and/or not numeric
        
    """
    pass


def construct_2x2(
    x: list[Any],
    y: list[Any],
    stringent: bool = False,
) -> np.ndarray:
    """Construct 2x2 metrics from list of x (i.e. condition) and y (i.e. outcome) data.
    
    Args:
        x: binary conditions (i.e. treated vs. not treated)
        y: binary outcomes (i.e. alive vs. dead)

    Raises:
        ValueError when more than 2 conditions are present in x and y
        ValueError when len of x and y differs after removing NAs
        ValueError when NAs present with the stringent set to True

    """
    pass


def dhyper(
    k: list[int],
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
) -> float:
    """Compute non-central hypergeometric distribution H with non-centrality parameter ncp, the odd ratio.
    
    Does not work for boundary values for ncp (0, int), but it does not need to.

    mapping of R to scipy::

        * def hyper_logpmf(k, M, n, N): return rmath.lib.dhyper(k, n, M-n, N, True)
        * def hyper_pmf(k, M, n, N): return rmath.lib.dhyper(k, n, M-n, N, False)
        * def hyper_cdf(k, M, n, N): return rmath.lib.phyper(k, n, M-n, N, True, False)
        * def hyper_sf(k, M, n, N): return rmath.lib.phyper(k, n, M-n, N, False, False)

    Args:
        k: # of Successes
        M: Total number of objects
        n: Total number of Type I objects (Total Positives)
        N: # of Total Type I object drawn (True Positive)
        is_log:

    Examples:
        >>> dhyper([0, 1, 2, 3], 10, 3, 4)  # 2x2 in [[1, 3], [2, 4]]
        array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738])
    
    """
    if is_log:
        return hypergeom.logpmf(k, M, n, N)

    return hypergeom.pmf(k, M, n, N)


def compute_dnhyper(
    support: list[int],
    logdc: list[float],
    ncp: int,
) -> float:
    """Compute ...
    
    Args:
        ncp: non-centrality parameter, the oddratio
    
    >>> compute_dnhyper(np.array([0, 1, 2, 3]), np.array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738]), 10)
    array([0.00243309, 0.0729927 , 0.4379562 , 0.486618  ])

    """
    d = logdc + np.log(ncp) * support
    d = np.exp(d - max(d))
    return d / np.sum(d)


def compute_mnhyper(
    support: list[int],
    data: np.ndarray,
    odd_ratio: int,
    is_log: bool=True,
):
    """Compute mnhyper.
    
    Args:
        support:
        data:
        odd_ratio:
        is_log: 
    
    Examples:
    >>>

    """
    return np.sum(support * compute_dnhyper(data, odd_ratio, is_log))



def compute_dnhyper(
    data: np.ndarray,
    odd_ratio: float | None = None,
    is_log: bool = True,
) -> float:
    """Compute non-central hypergeomtric distribution parameter."""
    mn = data.sum(axis=0)
    m = mn[0]  # Total Type I drawn (FP + TP)
    n = mn[1]  # FN + TN
    k = data.sum(axis=1)[0] # TP + FN

    x = data[0][0]
    lo = max(0, k - n)
    hi = min(k, m)
    nval = "odd_ratio"

    logdc = dhyper(np.arange(lo, hi + 1), m + n, k, m, log=is_log)



def fisher_exact() -> None:
    pass

