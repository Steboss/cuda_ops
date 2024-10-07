#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from cuda_ops import rms_norm


@pytest.mark.parametrize(
    "n",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
)
def test_rms_norm(n: int):
    """General test on RMS norm.
    This test checks whethe the CPU result and GPU are aligned"""
    m = np.random.randn(n, n).astype(np.float32)
    result_gpu = m.copy()
    rms_norm(result_gpu)  # in-place update
    result_cpu = m / np.sqrt(np.mean(m**2, axis=1))[:, None]

    assert np.allclose(result_cpu, result_gpu)


def test_rms_norm_with_known_input():
    """This is a test with a known input
    We can compute the result by hand here"""
    m = np.array([[3.0, 4.0]], dtype=np.float32)
    expected_rms = np.sqrt((9 + 16) / 2)
    expected_rms_norm = m / expected_rms

    rms_norm(m)

    assert np.allclose(m, expected_rms_norm, atol=1e-6)


def test_rms_norm_with_zero_matrix():
    """This is a test with a zero matrix"""
    m = np.zeros((2, 2), dtype=np.float32)

    try:
        rms_norm(m)
    except ZeroDivisionError:
        pass
    else:
        assert np.allclose(m, np.zeros_like(m))
