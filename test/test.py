#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from cuda_ops import rms_norm


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_rms_norm(n):
    m = np.random.randn(n, n)

    result_gpu = m.copy()
    rms_norm(result_gpu)  # in-place update
    result_cpu = m / np.sqrt(np.mean(m**2, axis=1))[:, None]

    assert np.allclose(result_cpu, result_gpu)
