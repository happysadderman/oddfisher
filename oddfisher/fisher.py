# oddfisher.py, a python version of R fisher exact test with odd ratio parameter.

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


def compute_mnhyper(
    support: list[int],
    M: int,
    n: int,
    N: int,
    odd_ratio: int | float = 1,
) -> float:
    """Compute mnhyper.
    
    Args:
    
    Examples:
    >>>

    """
    if odd_ratio == 0:
        return max(0, N - M + n)
    elif odd_ratio == np.inf:
        return min(N, n)
    return np.sum(support * compute_dnhyper(support, M, n, N, odd_ratio=odd_ratio))


def compute_pnhyper(
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    is_lower_tail: bool = True,
    odd_ratio: int | float = 1,
) -> float:
    """Compute phyper.
    
    Args:
    
    Returns:
    
    """
    lo = max(0, N - M + n)
    hi = min(N, n)

    if odd_ratio == 1:
        if not is_lower_tail:  # Upper tail = True
            x = x - 1
        return phyper(
            x,
            M,
            n,
            N,
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
    odd_ratio: int | float = 1.0,
) -> list[float]:
    """Compute non-central hypergeomtric distribution parameter.
    
    Args:
        support:
        M:
        n:
        N:
        odd_ratio:
        
    Returns:

    Examples:
        >>> support = np.arange(0, 4)
        >>> compute_dnhyper(support, 10, 3, 4, 10)
        array([0.00243309, 0.0729927 , 0.4379562 , 0.486618  ])
    """
    print(dhyper(support, M, n, N))
    d = dhyper(support, M, n, N) + np.log(odd_ratio) * support
    d = np.exp(d - max(d))
    return d / np.sum(d)


def get_pvalue(
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    odd_ratio: int | float,
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
        d = compute_dnhyper(support, M, n, N, odd_ratio=odd_ratio)
        print("d", d)
        two_tailed_val = sum(d[d <= d[x - lo + 1] * relError])
    
    lower_tail_val = compute_pnhyper(
        support,
        x,
        M,
        n,
        N,
        is_lower_tail=True,
        odd_ratio=odd_ratio,
    )

    upper_tail_val = compute_pnhyper(
        support,
        x,
        M,
        n,
        N,
        is_lower_tail=False,
        odd_ratio=odd_ratio,
    )

    return two_tailed_val, lower_tail_val, upper_tail_val


def get_confidence_interval(
    confidence_level: float,
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    odd_ratio: int | float,
    alternative: str, 
) -> tuple[float, float]:
    """Get confidence interval for the odd_ratio."""
    if alternative == "less":
        ncp_u = get_ncp_u(1 - confidence_level, support, x, M, n, N)
        return 0, ncp_u

    elif alternative == "greater":
        ncp_l = get_ncp_l(1 - confidence_level, support, x, M, n, N)
        return ncp_l, np.inf
    
    alpha = (1 - confidence_level) / 2
    return get_ncp_l(alpha, support, x, M, n, N), get_ncp_u(alpha, support, x, M, n, N)

def get_ncp_u(
    alpha,
    support,
    x,
    M,
    n,
    N,
):
    if x == min(N, n):
        return np.inf
    
    p = compute_pnhyper(support, x, M, n, N, odd_ratio=1, is_lower_tail=True)
    if p < alpha:
        return brentq(lambda t: compute_pnhyper(support, x, M, n, N, odd_ratio=t, is_lower_tail=True) - alpha, 0, 1)
    elif p > alpha:
        return 1 / brentq(lambda t: compute_pnhyper(support, x, M, n, N, odd_ratio=1/t, is_lower_tail=True) - alpha, np.finfo(float).eps, 1)
    else:
        return 1

def get_ncp_l(
    alpha,
    support,
    x,
    M,
    n,
    N,
):
    if x == max(0, N - M + n):
        return 0

    p = compute_pnhyper(support, x, M, n, N, odd_ratio=1, is_lower_tail=False)
    print("P from ncp_l", p, "alpha", alpha)
    if p > alpha:
        return brentq(lambda t: compute_pnhyper(support, x, M, n, N, odd_ratio=t, is_lower_tail=False) - alpha, 0, 1)
    elif p < alpha:  
        return 1 / brentq(lambda t: compute_pnhyper(support, x, M, n, N, odd_ratio=1/t, is_lower_tail=False) - alpha, np.finfo(float).eps, 1)
    else:
        return 1


def compute_mle_for_oddratio(
    support: list[int],
    x: int,
    M: int,
    n: int,
    N: int,
    odd_ratio: int | float,   
) -> int | float:
    """Compute MLE for odd ratio by solving E(X) = x."""
    lo = max(0, N - M + n)
    hi = min(N, n)

    if x == lo:
        return 0
    elif x == hi:
        return np.inf
    
    mu = compute_mnhyper(support, M, n, N, odd_ratio=1)
    print("mu", mu, "x", x)
    if mu > x:
        root = brentq(lambda t: compute_mnhyper(support, M, n, N, odd_ratio=t) - x, 0, 1)
    elif mu < x:
        root = brentq(lambda t: compute_mnhyper(support, M, n, N, odd_ratio=1/t) - x, np.finfo(float).eps, 1)
        root = 1 / root
    else:
        root = 1
    print("root", root)
    return root


def run_fisher_exact(
    data: np.ndarray,
    odd_ratio: int | float = 1,
    conf_level: float = 0.95,
    alternative: str = "two_sided",
) -> None:
    """
    >>> compute_dnhyper(np.array([[1, 3], [2, 4]]), odd_ratio=10, is_log=True)
    array([0.00243309, 0.0729927 , 0.4379562 , 0.486618  ])
    """
    mn = data.sum(axis=1)
    M = sum(mn)
    n = mn[1]  # Total diseased, TP + FN
    # M_minus_n = mn[0]  # Total healthy, FP + TN
    N = data.sum(axis=0)[0]  # Number called diseased, TP + FP

    x = data[0][0]  # TP
    lo = max(0, N - n)
    hi = min(N, n)
    # nval = "odd_ratio"
    support = np.arange(lo, hi)
    
    estimate = compute_mle_for_oddratio(support, x, M, M - n, N, odd_ratio=odd_ratio)
    # compute_dnhyper(support, M, n, N, is_log=is_log, odd_ratio=odd_ratio)
    print("Estimate", estimate)
    confidence_interval = get_confidence_interval(
        conf_level,
        support,
        x,
        M,
        M - n,
        N,
        odd_ratio=odd_ratio,
        alternative=alternative,
    )

    return estimate, confidence_interval


def arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='Commands', dest='command')
    cmd = commands.add_parser("fisherexact")
    
    cmd.add_argument("a", type=int, help="2x2 contingency table, a")
    cmd.add_argument("b", type=int, help="2x2 contingency table, b")
    cmd.add_argument("c", type=int, help="2x2 contingency table, c")
    cmd.add_argument("d", type=int, help="2x2 contingency table, d")
    cmd.add_argument("--alternative", type=str, default="two-sided", help="alternative hypothesis: one of two-sided (default), less, or greater")
    cmd.add_argument("--odd-ratio", type=float, default=1, help="hypothesized odd ratio")
    cmd.add_argument("--conf-level", type=float, default=0.95, help="confidence level, default is 0.95")
    cmd.set_defaults(func=run_fisher_exact)
    return parser


def main():
    args = arg_parser().parser()
    pval, ci = run_fisher_exact(
        data = np.array([args.a, args.b, args.c, args.d]).reshape((2, 2)),
        odd_ratio = args.odd_ratio,
        conf_level=args.conf_level,
        alternative=args.alternative,
    )

    print("2x2 contingency table")
    print(f"{args.a}, {args.b}")
    print(f"{args.c}, {args.d}")
    print("-------------------")
    print(f"odd-raio: {args.odd_ratio}")
    print(f"alternative: {args.alternative}")
    print("-------------------")
    print(f"p-value: {pval}")
    print(f"confidence interval at {args.conf_level}: {ci}")



def cli(*, argv: list[str] | None = None, args: argparse.Namespace | None = None) -> None:
    """Command line interface.

    Args:
        argv: command line parameters as an unparsed list of strings
        args: command line parameters as a parsed argparse.Namespace

    """
    parser = arg_parser()

    if argv is not None and args is not None:
        raise ValueError('argv and args are mutually exclusive')
    elif args is None:
        args = parser.parse_args(argv)

    if args.command and args.func:
        if args.command == "fisherexact":
            data = np.array([args.a, args.b, args.c, args.d]).reshape((2, 2))
            args.func(
                data=data,
                odd_ratio=args.odd_ratio,
                conf_level=args.conf_level,
                alternative=args.alternative,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()