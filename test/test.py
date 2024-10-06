#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import pytest

from cuda_ops import rms_norm

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "n",
    [
        1,
        2,
    ],
)
def test_rms_norm(n):
    # we need to convert to np.float32 to align with CUDA
    # standard NumPy uses float64
    m = np.random.randn(n, n).astype(np.float32)
    logger.info(f"matrix m {m}")
    result_gpu = m.copy()
    logger.info(f"GPU matrix {result_gpu}")
    rms_norm(result_gpu)  # in-place update
    logger.info(f"GPU matrix after rms_norm {result_gpu}")
    result_cpu = m / np.sqrt(np.mean(m**2, axis=1))[:, None]
    logger.info(f"CPU matrix after rms_norm {result_cpu}")

    assert np.allclose(result_cpu, result_gpu)


def test_rms_norm_with_known_input():
    """This is a test with a known input"""
    m = np.array([[3.0, 4.0]], dtype=np.float32)
    expected_rms = np.sqrt((9 + 16) / 2)
    expected_rms_norm = m / expected_rms

    rms_norm(m)

    assert np.allclose(m, expected_rms_norm, atol=1e-6)
