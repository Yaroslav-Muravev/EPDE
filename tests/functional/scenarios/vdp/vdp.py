import os
import torch
import numpy as np
from datetime import datetime
import json
import pytest
from epde.interface.interface import EpdeSearch
from epde import TrigonometricTokens, GridTokens
from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SingleEquationComparison
from tests.functional.utils.timer import Timer

class VanDerPolTest(EquationTestTemplate):
    strategy = SingleEquationComparison()

    def __init__(self, foldername = "", noise_level = 0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u"]

    def correct_symbolic(self):
        return "-0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} + -0.0 = d^2u/dx0^2{power: 1.0}"

    def incorrect_symbolic(self):
        return '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    def load_data(self):
        return np.load(os.path.join(self.foldername, "vdp_data.npy"))

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "vdp" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    @staticmethod
    def make_additional_tokens():
        trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=0)
        grid_tokens = GridTokens(["x_0"], dimensionality=0, max_power=2)
        return [grid_tokens, trig_tokens]

    @staticmethod
    def noise_data(data, noise_level):
        return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

    def make_search(self):
        step = 0.05
        steps_num = 320
        t = np.arange(0., step * steps_num, step)

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=2,
            coordinate_tensors=[t],
            verbose_params={"show_iter_idx": True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=self.load_data(),
            variable_names=["u"],
            max_deriv_order=(2,),
            additional_tokens=self.make_additional_tokens(),
        )
        return epde_search_obj

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        data = self.load_data()
        noised_data = self.noise_data(data, self.noise_level)

        search_obj.set_moeadd_params(population_size=16, training_epochs=1)

        with Timer() as t:
            search_obj.fit(
                data=[noised_data],
                variable_names=["u"],
                max_deriv_order=(2, 3),
                equation_terms_max_number=5,
                data_fun_pow=3,
                additional_tokens=self.make_additional_tokens(),
                equation_factors_max_number={"factors_num": [1, 2], "probas": [0.65, 0.35]},
                eq_sparsity_interval=(1e-5, 1e-0),
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)

            report = {
                "scenario": "VanDerPol",
                "operator": operator_name,
                "noise_level": self.noise_level,
                "elapsed_sec": t.elapsed,
            }
            (report_dir / "summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            raw_clusters = search_obj.equations(only_print=False, num=5)

            clusters_json = []
            for idx, cluster in enumerate(raw_clusters):
                equations_list = []
                for eq in cluster:
                    text = eq.text_form
                    if "\n" in text:
                        eq_part, meta_part = text.split("\n", 1)
                    else:
                        eq_part, meta_part = text, None

                    equations_list.append({
                        "equation": eq_part.strip(),
                        "parameters": meta_part
                    })
                clusters_json.append({
                    "cluster_id": idx,
                    "size": len(equations_list),
                    "equations": equations_list
                })

            # Сохраняем как JSON
            (report_dir / "equations.json").write_text(
                json.dumps(clusters_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return search_obj, t.elapsed