"""Data adapter for KdV (SINDy benchmark). See configs/kdv.yaml."""

import os
import numpy as np
from scipy.io import loadmat

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'kdv'
))


def load_data():
    d = loadmat(os.path.join(_DATA_DIR, 'kdv_sindy.mat'))
    t = np.ravel(d['t'])
    x = np.ravel(d['x'])
    u = np.transpose(np.real(d['usol']))
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), u, ['u'], 1
