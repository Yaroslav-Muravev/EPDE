"""Data adapter for Lotka-Volterra. See configs/lv.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'lv'
))


def load_data():
    t = np.load(os.path.join(_DATA_DIR, 't_20.npy'))[:150]
    data = np.load(os.path.join(_DATA_DIR, 'data_20.npy'))[:150]
    return (t,), [data[:, 0], data[:, 1]], ['u', 'v'], 0
