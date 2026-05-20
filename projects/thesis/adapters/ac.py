"""Data adapter for Allen-Cahn. See configs/ac.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'ac'
))


def load_data():
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(os.path.join(_DATA_DIR, 'ac_data.npy'))
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
