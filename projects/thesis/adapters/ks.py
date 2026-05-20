"""Data adapter for Kuramoto-Sivashinsky. See configs/ks.yaml."""

import os
import numpy as np
import scipy.io as scio

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'ks'
))


def load_data():
    d = scio.loadmat(os.path.join(_DATA_DIR, 'kuramoto_sivishinky.mat'))
    t = np.ravel(d['tt'])
    x = np.ravel(d['x'])
    u = d['uu'].T
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), u, ['u'], 1
