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
    """Compute non-central hypergeometric density distribution H with non-centrality parameter ncp, the odd ratio.
    
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
        is_log: True if in log scale

    Examples:
        >>> dhyper([0, 1, 2, 3], 10, 3, 4)  # 2x2 in [[1, 3], [2, 4]]
        array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738])
    
    """
    return hypergeom.logpmf(k, M, n, N) if is_log else hypergeom.pmf(k, M, n, N)


def phyper(
    k: int,
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
    is_lower_tail: bool = True,
) -> float:
    """Compute hypergeometric distribution H.
    
    Args:
        k: # of Successes (or called diseased)
        M: Total number of objects (TP + FN + FN + TN)
        n: Total number of Type I objects or has disease (Total Positives, TP + FN)
        N: # of Total Type I object drawn or dignosed as diseased (TP + FP)
        is_log: True if in log scale
        is_lower_tail: True if probabilities are P[X≤x], otherwise, P[X>x]

    Examples:
        >>> phyper([0, 1, 2, 3], 10, 3, 4, is_log=True, is_lower_Tail=False)  # 2x2 in [[1, 3], [2, 4]]
        array([-0.18232156, -1.09861229, -3.40119738,        -inf])
        >>> phyper([0, 1, 2, 3], 10, 3, 4, is_log=True, is_lower_Tail=True)
        array([-1.79175947, -0.40546511, -0.03390155,  0.        ])
        >>> phyper([0, 1, 2, 3], 10, 3, 4, is_log=False, is_lower_Tail=False)
        array([0.83333333, 0.33333333, 0.03333333, 0.        ])
        >>> phyper([0, 1, 2, 3], 10, 3, 4, is_log=False, is_lower_Tail=True)
        array([0.16666667, 0.66666667, 0.96666667, 1.        ])

    """
    if is_log:
        if is_lower_tail:
            return np.log(1 - np.exp(hypergeom.logsf(k, M, n, N)))
        else:
            return hypergeom.logsf(k, M, n, N)
    else:
        if is_lower_tail:
            return 1 - hypergeom.sf(k, M, n, N)
        else:
            return hypergeom.sf(k, M, n, N)
    

def delete_compute_dnhyper(
    support: list[int],
    logdc: list[float],
    odd_ratio: int | float,
) -> float:
    """Compute ...
    
    Args:
        odd_ratio: non-centrality parameter, the oddratio
    
    >>> compute_dnhyper(np.array([0, 1, 2, 3]), np.array([-1.79175947, -0.69314718, -1.2039728 , -3.40119738]), 10)
    array([0.00243309, 0.0729927 , 0.4379562 , 0.486618  ])

    """
    d = logdc + np.log(odd_ratio) * support
    d = np.exp(d - max(d))
    return d / np.sum(d)


def compute_mnhyper(
    support: list[int],
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
    odd_ratio: int | float = 1,
) -> float:
    """Compute mnhyper.
    
    Args:
    
    Examples:
    >>>

    """
    if odd_ratio == 0:
        return max(0, N - n)
    elif odd_ratio == np.inf:
        return min(N, n)
    return np.sum(support * compute_dnhyper(support, M, n, N, is_log=is_log, odd_ratio=odd_ratio))


def compute_pnhyper(
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
    is_lower_tail: bool = True,
    odd_ratio: int | float = 1,
) -> np.float:
    """Compute phyper.
    
    Args:
    
    Returns:
    
    """
    lo = max(0, N - n)
    hi = min(N, n)

    if odd_ratio == 1:
        if not is_lower_tail:
            x = x - 1
        return phyper(
            x,
            M,
            n,
            N,
            is_log=is_log,
            is_lower_tail=is_lower_tail,
        )
    
    if odd_ratio == 0:
        return int(x >= lo if is_lower_tail else x <= lo)
    
    if odd_ratio == np.inf:
        return int(x >= hi if is_lower_tail else x <= hi)
    
    return sum(
        compute_dnhyper(
            support,
            M,
            n,
            N,
            is_log=is_log,
            odd_ratio=odd_ratio,
        ) * [
            support <= x
        ] if is_lower_tail else [
            support >= x
        ]
    )


def compute_dnhyper(
    support: list[int],
    M: int,
    n: int,
    N: int,
    is_log: bool = True,
    odd_ratio: int | float = 1,
) -> list[float]:
    """Compute non-central hypergeomtric distribution parameter.
    
    """
    d = dhyper(support, M, n, N, log=is_log) + np.log(odd_ratio) * support
    d = np.exp(d - max(d))
    return d / np.sum(d)


def get_pvalue(
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    odd_ratio: int | float,
    is_log: bool,
    relError: float = 1 + 10 ** -7,
) -> tuple[int | float]:
    """Get p-values."""
    lo = max(0, N - n)
    hi = min(N, n)

    if odd_ratio == 0:
        two_tailed_val = int(x == lo)
    elif odd_ratio == np.inf:
        two_tailed_val = int(x == hi)
    else:
        d = compute_dnhyper(support, M, n, N, is_log=is_log, odd_ratio=odd_ratio)
        two_tailed_val = sum(d[d <= d[x - lo + 1] * relError])
    
    lower_tail_val = compute_pnhyper(
        support,
        x,
        M,
        n,
        N,
        is_log=is_log,
        is_lower_tail=True,
        odd_ratio=odd_ratio,
    )

    upper_tail_val = compute_pnhyper(
        support,
        x,
        M,
        n,
        N,
        is_log=is_log,
        is_lower_tail=False,
        odd_ratio=odd_ratio,
    )

    return two_tailed_val, lower_tail_val, upper_tail_val



def fisher_exact(
    data: np.ndarray,
    odd_ratio: int | float = 1,
    is_log: bool = True 
) -> None:
    """
    >>> compute_dnhyper(np.array([[1, 3], [2, 4]]), odd_ratio=10, is_log=True)
    array([0.00243309, 0.0729927 , 0.4379562 , 0.486618  ])
    """
    mn = data.sum(axis=0)
    M = sum(mn)
    n = mn[0]  # Total diseased, TP + FN
    # M_minus_n = mn[1]  # Total healthy, FP + TN
    N = data.sum(axis=1)[0]  # Number called diseased, TP + FP

    # x = data[0][0]  # TP
    lo = max(0, N - n)
    hi = min(N, n)
    # nval = "odd_ratio"
    support = np.arange(lo, hi + 1)
    
    # compute_dnhyper(support, M, n, N, is_log=is_log, odd_ratio=odd_ratio)

