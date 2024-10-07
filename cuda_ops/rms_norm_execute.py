#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np

from .rms_norm import rms_norm

logger = logging.getLogger(__name__)


def compute(matrix: np.array):
    """This function performs a data conversion over the given NumPy matrix

    Args:
        matrix (np.array): The input matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("The input matrix should be a NumPy array")

    if matrix.ndim != 2:
        raise ValueError("The input matrix should be a 2D matrix")

    if matrix.dtype != np.float64 and matrix.dtype != np.float32:
        raise ValueError("The input matrix should be of type float32 or float64")

    if matrix.dtype == np.float32:
        logger.info("Executing calculation with float32 precision")
    else:
        logger.info("Executing calculation with float64 precision")

    rms_norm(matrix)
