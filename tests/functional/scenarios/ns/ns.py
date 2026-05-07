import os
import json
import torch
from datetime import datetime
import numpy as np
import scipy.io as scio
import pytest
from epde.interface.interface import EpdeSearch
from tests.functional.templates import EquationTestTemplate
from tests.functional.comparasion import SystemComparison
from tests.functional.utils.timer import Timer

class NavierStokesTest(EquationTestTemplate):
    strategy = SystemComparison()

    def __init__(self, foldername="", noise_level=0):
        if foldername == "":
            foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.foldername = foldername
        self.noise_level = noise_level

    def all_vars(self):
        return ["u", "v", "p"]

    def correct_symbolic(self):
        eq_u_correct = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
        eq_v_correct = eq_u_correct.replace('u', 'v').replace('du/dx0', 'dv/dx0').replace('d^2u/dx1^2', 'd^2v/dx1^2')
        eq_p_correct = eq_u_correct.replace('u', 'p').replace('du/dx0', 'dp/dx0').replace('d^2u/dx1^2', 'd^2p/dx1^2')
        return [eq_u_correct, eq_v_correct, eq_p_correct]

    def incorrect_symbolic(self):
        eq_u_incorrect = '4.976781518840499 * u{power: 1.0} + 0.0001 * d^2u/dx1^2{power: 1.0} + -4.974425220166616 * u{power: 3.0} + 0.0 * du/dx1{power: 1.0} * d^2u/dx0^2{power: 1.0} + 0.002262543822130977 = du/dx0{power: 1.0}'
        eq_v_incorrect = eq_u_incorrect.replace('u', 'v').replace('du/dx0', 'dv/dx0').replace('d^2u/dx1^2', 'd^2v/dx1^2').replace('du/dx1', 'dv/dx1')
        eq_p_incorrect = eq_u_incorrect.replace('u', 'p').replace('du/dx0', 'dp/dx0').replace('d^2u/dx1^2', 'd^2p/dx1^2').replace('du/dx1', 'dp/dx1')
        return [eq_u_incorrect, eq_v_incorrect, eq_p_incorrect]

    def ns_data(self):
        """Загружает данные из .mat файла и формирует сетки."""
        mat = scio.loadmat(os.path.join(self.foldername, 'cylinder_nektar_wake.mat'))
        U_star = mat['U_star']
        P_star = mat['p_star']
        t_star = mat['t']
        X_star = mat['X_star']

        N = X_star.shape[0]
        T = t_star.shape[0]
        t_train = 50                    # как в ns_test

        x = np.unique(X_star[:, 0:1].flatten())
        y = np.unique(X_star[:, 1:2].flatten())
        t = t_star.flatten()

        u = U_star[:, 0, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]
        v = U_star[:, 1, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]
        p = P_star.T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]

        grids = np.meshgrid(t[:t_train], y, x, indexing='ij')
        data = [u, v, p]
        return grids, data

    def noise_data(self, data, noise_level):
        return [d + noise_level * 0.01 * np.std(d) * np.random.normal(size=d.shape) for d in data]

    def make_additional_tokens(self):
        return []

    def make_report_dir(self, base_dir, operator_name):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / "ns" / operator_name / stamp
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _save_report(self, report_dir, operator_name, elapsed, search_obj):
        report = {
            "scenario": "NavierStokes",
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
        """Создаёт EpdeSearch для тестового режима (сравнение)."""
        grids, data = self.ns_data()
        t, y, x = grids
        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=10,
            coordinate_tensors=(t, y, x),
            verbose_params={'show_iter_idx': True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        epde_search_obj.create_pool(
            data=data,
            variable_names=['u', 'v', 'p'],
            max_deriv_order=(2,2,2),
            additional_tokens=[],
        )
        return epde_search_obj

    @pytest.mark.slow
    def run_discovery(self, search_obj, report_dir=None, operator_name="unknown"):
        """Режим поиска уравнений (discovery)."""
        grids, data = self.ns_data()
        noised_data = self.noise_data(data, self.noise_level)
        t, y, x = grids

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=[21,21,46],
            coordinate_tensors=(t, y, x),
            verbose_params={'show_iter_idx': True},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})

        popsize = 64
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

        with Timer() as tim:
            epde_search_obj.fit(
                data=noised_data,
                variable_names=['u', 'v', 'p'],
                max_deriv_order=(1,2,2),
                equation_terms_max_number=20,
                data_fun_pow=1,
                additional_tokens=[],
                equation_factors_max_number=factors_max_number,
                eq_sparsity_interval=(1e-12, 1e-0),
            )

        if report_dir is not None:
            report_dir = self.make_report_dir(report_dir, operator_name)
            self._save_report(report_dir, operator_name, tim.elapsed, epde_search_obj)

        return search_obj, tim.elapsed