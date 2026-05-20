"""Data adapter for Van der Pol. See configs/vdp.yaml.

build_extra_tokens supplies a tight TrigonometricTokens around freq=2 so the
search exposes ``sin(2t)`` as a factor even though the truth equation
doesn't actually use it (kept for parity with the LEGACY runner).
"""

import os
import numpy as np

from epde import TrigonometricTokens

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'vdp'
))


def load_data():
    step, n = 0.05, 320
    t = np.arange(0., step * n, step)
    data = np.load(os.path.join(_DATA_DIR, 'vdp_data.npy'))
    return (t,), [data], ['u'], 0


def build_extra_tokens(coords, dim):
    return [TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=dim)]
