"""Data adapter for the 1+1D wave equation. See configs/wave.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'wave'
))


def load_data():
    shape = 80
    data = np.loadtxt(os.path.join(_DATA_DIR, 'wave_sln_80.csv'), delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
