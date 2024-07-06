# oddfisher.py, a python version of R fisher exact test with od parameter.

import os
import sys
import argparse

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import hypergeom as dhyper
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


def compute_dnhyper(
    data: np.ndarray,
    odd_ratio: float | None = None,
    is_log: bool = True,
) -> float:
    """Compute non-central hypergeomtric distribution parameter."""
    mn = data.sum(axis=0)
    m = mn[0]
    n = mn[1]
    k = data.sum(axis=1)[0]

    x = data[0][0]
    lo = max(0, k - n)
    hi = min(k, m)
    nval = odd_ratio

    logdc = dhyper(np.arange(lo, hi + 1), m, n, k, log=is_log)


def fisher_exact() -> None:
    pass

