"""Data adapter for Forced Damped Oscillator. See configs/ode.yaml."""

import os
import numpy as np

from epde import TrigonometricTokens

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'ode'
))


def load_data():
    step, n = 0.05, 320
    t = np.arange(0., step * n, step)
    data = np.load(os.path.join(_DATA_DIR, 'ode_data.npy'))
    return (t,), [data], ['u'], 0


def build_extra_tokens(coords, dim):
    return [TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=dim)]
