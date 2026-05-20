"""Data adapter for Burgers viscous (SINDy nu=0.1). See configs/burgers_viscous.yaml."""

import os
import numpy as np
from scipy.io import loadmat

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'burgers'
))


def load_data():
    burg = loadmat(os.path.join(_DATA_DIR, 'burgers.mat'))
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    data = np.transpose(np.real(burg['usol']))
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
