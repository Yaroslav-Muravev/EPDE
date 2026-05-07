import os
import json
import pickle
from datetime import datetime
import torch
import pytest

import numpy as np
import pandas as pd
from scipy.io import loadmat

from epde.interface.interface import EpdeSearch
from epde import TrigonometricTokens, CacheStoredTokens

from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SingleEquationComparison
from tests.functional.utils.timer import Timer


def load_pretrained_PINN(ann_filename):
    try:
        with open(ann_filename, "rb") as data_input_file:
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print("No model located, proceeding without pretrained ANN.")
        data_nn = None
    return data_nn


class BurgersTest(EquationTestTemplate):
    strategy = SingleEquationComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u"]

    def correct_symbolic(self):
        # Оставил в стиле вашего текущего сценария.
        # Если это не истинная формула Burgers, замените на нужную.
        return '-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.01 * d^2u/dx1^2{power: 1.0} + 0.0 = du/dx0{power: 1.0}'

    def incorrect_symbolic(self):
        return '0.02 * d^2u/dx1^2{power: 1.0} + -0.98 * u{power: 1.0} * du/dx1{power: 1.0} + 0.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'

    @staticmethod
    def noise_data(data, noise_level):
        return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "burgers" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _save_report(self, report_dir, operator_name, elapsed, search_obj):
        report = {
            "scenario": "Burgers",
            "operator": operator_name,
            "noise_level": self.noise_level,
            "elapsed_sec": elapsed,
        }
        (report_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raw_clusters = search_obj.equations(only_print=False, num=5)

        def eq_to_text(eq):
            return getattr(eq, "text_form", str(eq))

        clusters_json = []
        for idx, cluster in enumerate(raw_clusters):
            equations_list = []
            for eq in cluster:
                equations_list.append({"equation": eq_to_text(eq)})
            clusters_json.append(
                {
                    "cluster_id": idx,
                    "size": len(equations_list),
                    "equations": equations_list,
                }
            )

        (report_dir / "equations.json").write_text(
            json.dumps(clusters_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_burgers_sindy_data(self, filename):
        burg = loadmat(filename)
        t = np.ravel(burg["t"])
        x = np.ravel(burg["x"])
        data = np.real(burg["usol"])
        data = np.transpose(data)
        grids = np.meshgrid(t, x, indexing="ij")
        return grids, data

    def load_burgers_csv_data(self, filename):
        df = pd.read_csv(filename, header=None)
        u = df.values
        data = np.transpose(u)
        t = np.linspace(0, 1, 101)
        x = np.linspace(-1000, 0, 101)
        grids = np.meshgrid(t, x, indexing="ij")
        return grids, data

    def make_additional_tokens_sindy(self):
        return []

    def make_additional_tokens_discovery(self, grid):
        custom_grid_tokens = CacheStoredTokens(
            token_type="grid",
            token_labels=["t", "x"],
            token_tensors={"t": grid[0], "x": grid[1]},
            params_ranges={"power": (1, 1)},
            params_equality_ranges=None,
        )
        trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq=(0.999, 1.001))
        return [custom_grid_tokens]

    def make_search(self):
        grid, data = self.load_burgers_sindy_data(os.path.join(self.foldername, "burgers.mat"))

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=(grid[0], grid[1]),
            verbose_params={"show_iter_idx": True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 2),
            additional_tokens=self.make_additional_tokens_sindy(),
        )
        return epde_search_obj

    @pytest.mark.slow
    def run_sindy_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        grid, data = self.load_burgers_sindy_data(os.path.join(self.foldername, "burgers.mat"))
        noised_data = self.noise_data(data, self.noise_level)

        search_obj.set_moeadd_params(population_size=16, training_epochs=15)

        with Timer() as t:
            search_obj.fit(
                data=noised_data,
                variable_names=["u"],
                max_deriv_order=(2, 3),
                derivs=None,
                equation_terms_max_number=5,
                data_fun_pow=3,
                additional_tokens=[],
                equation_factors_max_number={"factors_num": [1, 2], "probas": [0.65, 0.35]},
                eq_sparsity_interval=(1e-5, 1e2),
                fourier_layers=False,
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, t.elapsed, search_obj)

        return search_obj, t.elapsed

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        grid, data = self.load_burgers_csv_data(os.path.join(self.foldername, "burgers_sln_100.csv"))
        noised_data = self.noise_data(data, self.noise_level)

        search_obj.set_moeadd_params(population_size=16, training_epochs=2)

        with Timer() as t:
            search_obj.fit(
                data=noised_data,
                variable_names=["u"],
                max_deriv_order=(2, 3),
                derivs=None,
                equation_terms_max_number=5,
                data_fun_pow=3,
                additional_tokens=self.make_additional_tokens_discovery(grid),
                equation_factors_max_number={"factors_num": [1, 2], "probas": [0.65, 0.35]},
                eq_sparsity_interval=(1e-5, 1e2),
                fourier_layers=False,
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, t.elapsed, search_obj)

        return search_obj, t.elapsed