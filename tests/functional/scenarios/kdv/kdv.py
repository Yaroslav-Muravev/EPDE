import os
import json
from datetime import datetime
import scipy.io as scio

import numpy as np
import torch
import pytest
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde.interface.interface import EpdeSearch
from epde import TrigonometricTokens
from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SingleEquationComparison
from tests.functional.utils.timer import Timer


class KdVTest(EquationTestTemplate):
    strategy = SingleEquationComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u"]

    def correct_symbolic(self):
        return '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                           1.0 * cos(t)sin(x){power: 1.0} + \
                           0.0 = du/dx0{power: 1.0}'

    def incorrect_symbolic(self):
        return '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

    def load_data(self):
        filename = os.path.join(self.foldername, "data.csv")
        data = np.loadtxt(filename, delimiter=",").T
        shape = 80
        t = np.linspace(0, 1, shape + 1)
        x = np.linspace(0, 1, shape + 1)
        grids = np.meshgrid(t, x, indexing="ij")
        return grids, data

    def load_kdv_sindy_data(self):
        filename = os.path.join(self.foldername, "kdv_sindy.mat")
        data = scio.loadmat(filename)
        t = np.ravel(data["t"])
        x = np.ravel(data["x"])
        u = np.real(data["usol"])
        u = np.transpose(u)
        grids = np.meshgrid(t, x, indexing="ij")
        return grids, u

    @staticmethod
    def noise_data(data, noise_level):
        return noise_level * np.std(data) * np.random.normal(size=data.shape) + data

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "kdv" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def make_custom_tokens(self):
        custom_trigonometric_eval_fun = {
            "cos(t)sin(x)": lambda *grids, **kwargs: (
                np.cos(grids[0]) * np.sin(grids[1])
            ) ** kwargs["power"]
        }
        custom_trig_evaluator = CustomEvaluator(
            custom_trigonometric_eval_fun,
            eval_fun_params_labels=["power"],
        )
        return CustomTokens(
            token_type="trigonometric",
            token_labels=["cos(t)sin(x)"],
            evaluator=custom_trig_evaluator,
            params_ranges={"power": (1, 1)},
            params_equality_ranges={},
            meaningful=True,
            unique_token_type=True,
        )

    def _save_report(self, report_dir, operator_name, elapsed, search_obj):
        report = {
            "scenario": "KdV",
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

    def make_search(self):
        grid, data = self.load_data()

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=(grid[0], grid[1]),
            verbose_params={"show_iter_idx": True},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 3),
            additional_tokens=[self.make_custom_tokens()],
        )
        return epde_search_obj

    def make_search_sindy(self):
        grid, data = self.load_kdv_sindy_data()

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=(40, 100),
            coordinate_tensors=grid,
            verbose_params={"show_iter_idx": True},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type="poly", preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 3),
            additional_tokens=[],
        )
        return epde_search_obj

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        _, data = self.load_data()
        noised_data = self.noise_data(data, self.noise_level)

        search_obj.set_moeadd_params(population_size=16, training_epochs=5)

        with Timer() as t:
            search_obj.fit(
                data=noised_data,
                variable_names=["u"],
                max_deriv_order=(2, 3),
                derivs=None,
                equation_terms_max_number=10,
                data_fun_pow=3,
                additional_tokens=[self.make_custom_tokens()],
                equation_factors_max_number={"factors_num": [1, 2], "probas": [0.65, 0.35]},
                eq_sparsity_interval=(1e-5, 1e-2),
                fourier_layers=False,
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, t.elapsed, search_obj)

        return search_obj, t.elapsed

    @pytest.mark.slow
    def run_sindy_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        grid, data = self.load_kdv_sindy_data()
        noised_data = self.noise_data(data, self.noise_level)

        search_obj.set_moeadd_params(population_size=16, training_epochs=1)

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
                eq_sparsity_interval=(1e-12, 1e-0),
                fourier_layers=False,
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name + "_sindy")
            self._save_report(report_dir, operator_name + "_sindy", t.elapsed, search_obj)

        return search_obj, t.elapsed