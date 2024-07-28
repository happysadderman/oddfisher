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
        k: # of Successes (or called diseased)
        M: Total number of objects (TP + FN + FN + TN)
        n: Total number of Type I objects or has disease (Total Positives, TP + FN)
        N: # of Total Type I object drawn or dignosed as diseased (TP + FP)
        is_log:

    Examples:
        >>> dhyper([0, 1, 2, 3], 10, 3, 4)  # 2x2 in [[1, 3], [2, 4]]
        array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738])
    
    """
    return hypergeom.logpmf(k, M, n, N) if is_log else hypergeom.pmf(k, M, n, N)


def phyper(
    k: list[int],
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
) -> float:
    

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


def compute_phyer(
    q: int,
    ncp: int = 1,
    use_upper_tail: bool = False,  
) -> np.float:
    """Compute phyper.
    
    Args:
    
    Returns:
    
    """
    if ncp == 1:
        return phyper(
            x - 1 if use_upper_tail else x,
            m,
            n,
            k,
            use_upper_tail=use_upper_tail,
        )
    
    if ncp == 0:
        return q <= lo if use_upper_tail else q >= lo
    
    if ncp == np.inf:
        return q <= hi if use_upper_tail else q >= hi
    
    return sum(dnhyper(ncp) * [support >= q] if use_upper_tail else [support <= q])


def compute_dnhyper(  # dhyper?
    data: np.ndarray,
    odd_ratio: float | None = None,
    is_log: bool = True,
) -> float:
    """Compute non-central hypergeomtric distribution parameter."""
    mn = data.sum(axis=0)
    M = sum(mn)
    n = mn[0]  # Total diseased, TP + FN
    # M_minus_n = mn[1]  # Total healthy, FP + TN
    N = data.sum(axis=1)[0]  # Number called diseased, TP + FP

    # x = data[0][0]  # TP
    lo = max(0, N - n)
    hi = min(N, n)
    nval = "odd_ratio"

    logdc = dhyper(np.arange(lo, hi + 1), M, n, N, log=is_log)



def fisher_exact() -> None:
    pass

