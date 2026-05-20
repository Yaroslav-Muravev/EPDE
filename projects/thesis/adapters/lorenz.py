"""Data adapter for Lorenz system. See configs/lorenz.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'lorenz'
))


def load_data():
    t = np.load(os.path.join(_DATA_DIR, 't.npy'))[:1000]
    data = np.load(os.path.join(_DATA_DIR, 'lorenz.npy'))[:1000]
    return (t,), [data[:, 0], data[:, 1], data[:, 2]], ['u', 'v', 'w'], 0
