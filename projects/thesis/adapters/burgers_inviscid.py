"""Data adapter for Burgers inviscid. See configs/burgers_inviscid.yaml."""

import os
import numpy as np
import pandas as pd

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'burgers'
))


def load_data():
    df = pd.read_csv(os.path.join(_DATA_DIR, 'burgers_sln_100.csv'), header=None)
    data = np.transpose(df.values)
    t = np.linspace(0, 1, 101)
    x = np.linspace(-1000, 0, 101)
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1
