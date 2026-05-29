"""Data adapter for Navier-Stokes (cylinder wake, Re=100).

Mirrors the data layout from ``projects/pic/data/ns/ns.py:ns_data``:
loads ``cylinder_nektar_wake.mat`` and reshapes ``U_star`` / ``p_star``
into ``(T, ny, nx)`` arrays restricted to the first ``T_TRAIN`` snapshots
for tractability (full series is ~200 frames; SINDy benchmarks usually
work with a 50-frame slice).
"""

import os
import numpy as np
from scipy.io import loadmat

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'ns'
))

# Time slice -- match the legacy ns.py reference (first 50 frames).
T_TRAIN = 50


def load_data():
    d = loadmat(os.path.join(_DATA_DIR, 'cylinder_nektar_wake.mat'))
    U_star = d['U_star']         # (N, 2, T) -- velocity components
    P_star = d['p_star']         # (N, T)    -- pressure
    t_star = d['t']              # (T, 1)
    X_star = d['X_star']         # (N, 2)

    x = np.unique(X_star[:, 0])
    y = np.unique(X_star[:, 1])
    t = t_star.ravel()

    u = U_star[:, 0, :].T.reshape(t.shape[0], y.shape[0], x.shape[0])[:T_TRAIN]
    v = U_star[:, 1, :].T.reshape(t.shape[0], y.shape[0], x.shape[0])[:T_TRAIN]
    p = P_star.T.reshape(t.shape[0], y.shape[0], x.shape[0])[:T_TRAIN]

    grids = np.meshgrid(t[:T_TRAIN], y, x, indexing='ij')
    return tuple(grids), [u, v, p], ['u', 'v', 'p'], 2
