"""Data adapter for KdV with cos(t)*sin(x) source. See configs/kdv_cossin.yaml."""

import os
import numpy as np

from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator

_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'kdv'
))


def load_data():
    shape = 80
    data = np.loadtxt(os.path.join(_DATA_DIR, 'data.csv'), delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')
    return tuple(grids), data, ['u'], 1


def build_extra_tokens(coords, dim):
    custom_eval = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']
    }
    evaluator = CustomEvaluator(custom_eval, eval_fun_params_labels=['power'])
    return [CustomTokens(
        token_type='trigonometric',
        token_labels=['cos(t)sin(x)'],
        evaluator=evaluator,
        params_ranges={'power': (1, 1)},
        params_equality_ranges={},
        meaningful=True,
        unique_token_type=False,
    )]
