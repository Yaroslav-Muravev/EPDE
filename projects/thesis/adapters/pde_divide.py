"""Data adapter for synthetic rational PDE. See configs/pde_divide.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'pde_divide'
))


def load_data():
    data = np.load(os.path.join(_DATA_DIR, 'PDE_divide.npy'))
    nx, nt = 100, 251
    x = np.linspace(1, 2, nx)
    t = np.linspace(0, 0.5, nt)
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
