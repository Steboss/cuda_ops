#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from cuda_ops import rms_norm_execute


@pytest.mark.parametrize(
    "n",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
)
def test_rms_norm(n: int):
    """General test on RMS norm.
    This test checks whethe the CPU result and GPU are aligned"""
    m = np.random.randn(n, n)
    result_gpu = m.copy()
    rms_norm_execute.compute(result_gpu)  # in-place update
    result_cpu = m / np.sqrt(np.mean(m**2, axis=1))[:, None]

    assert np.allclose(result_cpu, result_gpu)


def test_rms_norm_with_known_input():
    """This is a test with a known input
    We can compute the result by hand here"""
    m = np.array([[3.0, 4.0]])
    expected_rms = np.sqrt((9 + 16) / 2)
    expected_rms_norm = m / expected_rms

    rms_norm_execute.compute(m)

    assert np.allclose(m, expected_rms_norm, atol=1e-6)


def test_rms_norm_with_zero_matrix():
    """This is a test with a zero matrix"""
    m = np.zeros((2, 2))

    try:
        rms_norm_execute.compute(m)
    except ZeroDivisionError:
        pass
    else:
        assert np.allclose(m, np.zeros_like(m))


def test_rms_norm_large_values():
    """This is a test to see how large values are handled"""
    large_value = 1e20
    m = np.full((10, 10), large_value)
    result_gpu = m.copy()
    rms_norm_execute.compute(result_gpu)
    # we are expecting a matrix of 1s
    expected_output = np.ones_like(m)

    assert np.allclose(result_gpu, expected_output, atol=1e-6)


def test_rms_norm_small_values():
    """This is a test to see how small values are handled"""
    small_value = 1e-20
    m = np.full((10, 10), small_value)
    result_gpu = m.copy()
    rms_norm_execute.compute(result_gpu)
    # we are expecting a matrix of 1s
    expected_output = np.ones_like(m)

    assert np.allclose(result_gpu, expected_output, atol=1e-6)


def test_rms_norm_high_range():
    """This is a test where the matrix values are ranging from 10^-20 to 10^20"""
    exponents = np.linspace(-20, 20, 10)
    values = np.power(10, exponents)
    m = np.tile(values, (10, 1))
    result_gpu = m.copy()
    rms_norm_execute.compute(result_gpu)
    rms_cpu = np.sqrt(np.mean(m**2, axis=1))[:, None]
    expected_output = m / rms_cpu

    assert np.allclose(result_gpu, expected_output, atol=1e-6)


def test_rms_norm_1d_array():
    """This is a test to see if the checks are working with 1D arrays"""
    m = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="The input matrix should be a 2D matrix"):
        rms_norm_execute.compute(m)
