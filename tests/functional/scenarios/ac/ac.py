import os
import json
import pickle
import torch
import pytest
from datetime import datetime

import numpy as np
from epde.interface.interface import EpdeSearch

from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SingleEquationComparison
from tests.functional.utils.timer import Timer


class ACTest(EquationTestTemplate):
    strategy = SingleEquationComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u"]

    def correct_symbolic(self):
        return '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'

    def incorrect_symbolic(self):
        return ' 0.0001 * d^2u/dx1^2{power: 1.0} + -4.976781518840499 * u{power: 3.0} + 4.974425220166616 * u{power: 1.0} + 0.0 * du/dx1{power: 1.0} * d^2u/dx1^2{power: 1.0} + 0.002262543822130977 = du/dx0{power: 1.0}'

    def load_data(self):
        return np.load(os.path.join(self.foldername, "ac_data.npy"))

    def load_pretrained_PINN(self):
        ann_path = os.path.join(self.foldername, "ac_ann_pretrained.pickle")
        try:
            with open(ann_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("No model located, proceeding without pretrained ANN.")
            return None

    @staticmethod
    def noise_data(data, noise_level):
        return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

    def make_additional_tokens(self):
        return []

    def make_search(self):
        grid, data = self.ac_data()
        data_nn = self.load_pretrained_PINN()

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=(5, 12),
            coordinate_tensors=(grid[0], grid[1]),
            verbose_params={"show_iter_idx": True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 3),
            additional_tokens=self.make_additional_tokens(),
            data_nn=data_nn,
        )
        return epde_search_obj

    def ac_data(self):
        t = np.linspace(0.0, 1.0, 51)
        x = np.linspace(-1.0, 0.984375, 128)
        data = self.load_data()
        grids = np.meshgrid(t, x, indexing="ij")
        return grids, data

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "ac" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        _, data = self.ac_data()
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
                additional_tokens=self.make_additional_tokens(),
                equation_factors_max_number={"factors_num": [1, 2], "probas": [0.65, 0.35]},
                eq_sparsity_interval=(1e-12, 1e-0),
                fourier_layers=False,
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)

            report = {
                "scenario": "AllenCahn",
                "operator": operator_name,
                "noise_level": self.noise_level,
                "elapsed_sec": t.elapsed,
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
                    text = eq_to_text(eq)
                    equations_list.append({"equation": text})
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

        return search_obj, t.elapsed