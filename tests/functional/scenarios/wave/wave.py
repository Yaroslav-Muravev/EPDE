import os
import json
import torch
from datetime import datetime
import numpy as np
import pytest
from epde.interface.interface import EpdeSearch
from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SingleEquationComparison
from tests.functional.utils.timer import Timer

class WaveTest(EquationTestTemplate):
    strategy = SingleEquationComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u"]

    def correct_symbolic(self):
        # Волновое уравнение: d^2u/dt^2 = c^2 * d^2u/dx^2, c^2 = 0.04
        return '0.04 * d^2u/dx1^2{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}'

    def incorrect_symbolic(self):
        # Неправильное уравнение (близкое, с лишним множителем du/dx0)
        return '0.04 * d^2u/dx1^2{power: 1.0} * du/dx0{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0} * du/dx0{power: 1.0}'

    def wave_data(self):
        """Загружает данные из CSV и формирует сетки."""
        shape = 80
        data = np.loadtxt(os.path.join(self.foldername, 'wave_sln_80.csv'), delimiter=',').T
        t = np.linspace(0, 1, shape + 1)
        x = np.linspace(0, 1, shape + 1)
        grids = np.meshgrid(t, x, indexing='ij')
        return grids, data

    def noise_data(self, data, noise_level):
        return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

    def make_additional_tokens(self):
        return []

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "wave" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _save_report(self, report_dir, operator_name, elapsed, search_obj):
        report = {
            "scenario": "Wave",
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
            clusters_json.append({
                "cluster_id": idx,
                "size": len(equations_list),
                "equations": equations_list,
            })

        (report_dir / "equations.json").write_text(
            json.dumps(clusters_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def make_search(self):
        """Создаёт объект EpdeSearch для тестового режима (сравнение правильного/неправильного)."""
        grids, data = self.wave_data()
        t = grids[0]
        x = grids[1]

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,          # важно для совместимости с SingleEquationComparison
            use_pic=True,
            boundary=5,
            coordinate_tensors=(t, x),
            verbose_params={'show_iter_idx': True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=['u'],
            max_deriv_order=(2,2),
            additional_tokens=[],
        )
        return epde_search_obj

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        """Режим поиска уравнений."""
        grids, data = self.wave_data()
        noised_data = self.noise_data(data, self.noise_level)
        t = grids[0]
        x = grids[1]

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=20,
            coordinate_tensors=(t, x),
            verbose_params={'show_iter_idx': True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})

        popsize = 16
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

        with Timer() as tim:
            epde_search_obj.fit(
                data=noised_data,
                variable_names=['u'],
                max_deriv_order=(2,3),
                equation_terms_max_number=5,
                data_fun_pow=3,
                additional_tokens=[],
                equation_factors_max_number=factors_max_number,
                eq_sparsity_interval=(1e-6, 1e-4),
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, tim.elapsed, epde_search_obj)

        return search_obj, tim.elapsed