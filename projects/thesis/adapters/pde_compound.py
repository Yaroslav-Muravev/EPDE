"""Data adapter for synthetic compound PDE. See configs/pde_compound.yaml."""

import os
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'pde_compound'
))


def load_data():
    data = np.load(os.path.join(_DATA_DIR, 'PDE_compound.npy'))
    nx, nt = 100, 251
    x = np.linspace(1, 2, nx)
    t = np.linspace(0, 0.5, nt)
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
