import os
import json
import pickle
import torch
from datetime import datetime
import pytest
import numpy as np
from epde.interface.interface import EpdeSearch
from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SystemComparison
from epde import TrigonometricTokens, GridTokens
from tests.functional.utils.timer import Timer

class LorenzTest(EquationTestTemplate):
    strategy = SystemComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

        self.dimensionality = None

    def all_vars(self):
        return ["u", "v", "w"]

    def correct_symbolic(self):
        return [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} + 0.0 = dw/dx0{power: 1.0}'
    ]

    def incorrect_symbolic(self):
        return [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.1 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} + 0.1 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} + 0.1 * w{power: 1.0} + 0.0 = dw/dx0{power: 1.0}'
    ]

    def load_pretrained_PINN(self):
        lorenz_path = os.path.join(self.foldername, "lorenz_pretrained.pickle")
        try:
            with open(lorenz_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print('No model located, proceeding with ann approx. retraining.')
            return None

    def make_additional_tokens(self):
        return []

    @staticmethod
    def noise_data(data, noise_level):
        return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

    def make_additional_tokens(self):
        trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=self.dimensionality)
        grid_tokens = GridTokens(["x_0"], dimensionality=self.dimensionality, max_power=2)
        return [trig_tokens]

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "lorenz" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _save_report(self, report_dir, operator_name, elapsed, search_obj):
        report = {
            "scenario": "Lorenz",
            "operator": operator_name,
            "noise_level": self.noise_level,
            "elapsed_sec": elapsed,
        }
        (report_dir / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raw_clusters = search_obj.equations(only_print=False, num=1)

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
        t, data = self.lorenz_data()
        end = 1000
        t = t[:end]
        x = data[:end, 0]
        y = data[:end, 1]
        z = data[:end, 2]

        self.dimensionality = x.ndim - 1

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=(100,),
            coordinate_tensors=[t],
            verbose_params={'show_iter_idx': True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=[x, y, z],
            variable_names=['u', 'v', 'w'],
            max_deriv_order=1,
            additional_tokens=[],
        )
        return epde_search_obj

    def lorenz_data(self):
        t = np.load(os.path.join(os.path.dirname(__file__), 't.npy'))
        data = np.load(os.path.join(os.path.dirname(__file__), 'lorenz.npy'))
        return t, data

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        t, data = self.lorenz_data()
        noised_data = self.noise_data(data, self.noise_level)
        end = 1000
        t = t[:end]
        x = noised_data[:end, 0]
        y = noised_data[:end, 1]
        z = noised_data[:end, 2]

        epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=(100),
                                     coordinate_tensors=[t, ], verbose_params={'show_iter_idx': True},
                                     device='cuda' if torch.cuda.is_available() else 'cpu')

        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})

        popsize = 16
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

        with Timer() as tim:
            epde_search_obj.fit(data=[x, y, z], variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                                equation_terms_max_number=5, data_fun_pow=1, additional_tokens= self.make_additional_tokens(),
                                equation_factors_max_number=factors_max_number,
                                eq_sparsity_interval=(1e-8, 1e-0))

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, tim.elapsed, epde_search_obj)

        return search_obj, tim.elapsed

