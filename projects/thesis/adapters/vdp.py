"""Data adapter for Van der Pol. See configs/vdp.yaml.

The narrow ``TrigonometricTokens(freq=2 +/- 1e-8)`` previously declared
here is now in ``configs/defaults.yaml`` so every system shares the
same trig search space; this adapter only loads the raw signal.
"""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'vdp'
))


def load_data():
    step, n = 0.05, 320
    t = np.arange(0., step * n, step)
    data = np.load(os.path.join(_DATA_DIR, 'vdp_data.npy'))
    return (t,), [data], ['u'], 0
