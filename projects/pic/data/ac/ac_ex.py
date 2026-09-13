#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностический скрипт для проверки уравнений через SolverBasedFitness
(DeepXDE backend).

Запуск:
    python check_ac.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.objectives import Discrepancy, Instability
from epde.operators.common.fitness import SolverBasedFitness
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
import epde.globals as global_var
from epde.structure.main_structures import SoEq, Chromosome


# ----------------------------------------------------------------------
# Загрузка данных AC
# ----------------------------------------------------------------------
def ac_data(filename: str):
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename)
    grids = np.meshgrid(t, x, indexing='ij')
    return grids, data


def noise_data(data, noise_level):
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data


# ----------------------------------------------------------------------
# Набор проверяемых уравнений
# ----------------------------------------------------------------------
EQUATIONS_TO_CHECK = [
    (
        "CORRECT AC",
        '0.0001 * d^2u/dx1^2{power: 1.0} + '
        '-5.0 * u{power: 3.0} + '
        '5.0 * u{power: 1.0} + '
        '0.0 = du/dx0{power: 1.0}',
    ),
    (
        "DISCOVERED #1",
        '-0.06741029623854099 * d^2u/dx0^2{power: 1.0} + '
        '0.3316648582002735 * du/dx0{power: 1.0} + 0.0 = '
        'du/dx0{power: 1.0} * u{power: 2.0}',
    ),
    (
        "DISCOVERED #2",
        '4.968735087004665 * u{power: 1.0} + '
        '-4.968249064314862 * u{power: 3.0} + 0.0 = '
        'du/dx0{power: 1.0}',
    ),
    (
        "DISCOVERED #3",
        '3.31243377973145 * u{power: 2.0} * du/dx0{power: 1.0} + '
        '0.3382727271807673 * d^2u/dx0^2{power: 1.0} * u{power: 2.0} + 0.0 = '
        'du/dx0{power: 1.0}',
    ),
    (
        "DISCOVERED #4",
        '-0.599867874310047 * d^2u/dx0^2{power: 1.0} * u{power: 2.0} + '
        '0.27200598977275614 * d^2u/dx0^2{power: 1.0} + 0.0 = '
        'du/dx0{power: 1.0}',
    ),
]


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def check_equations(foldername: str, equations=None,
                    noise_level: float = 0.0):
    """
    Прогоняет список уравнений через SolverBasedFitness (DeepXDE)
    и печатает фитнес и стабильность для каждого.
    """
    if equations is None:
        equations = EQUATIONS_TO_CHECK

    # 1. Загрузка данных
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level) if noise_level > 0 else data

    print(f"[CHECK] data shape = {data.shape}, "
          f"range = [{data.min():.4f}, {data.max():.4f}], "
          f"std = {data.std():.4f}")

    # 2. Инициализация EpdeSearch и домена (один раз для всех уравнений)
    epde_search_obj = EpdeSearch(
        use_solver=True,
        verbose_params={'show_iter_idx': False},
        device='cpu',
    )
    _, domain = epde_search_obj.createDomain(
        (grid[0], grid[1]),
        boundary_width=(5, 12),
        ID=0,
    )
    epde_search_obj.set_preprocessor(
        default_preprocessor_type='FD',
        preprocessor_kwargs={},
    )
    _, trajectory = epde_search_obj.createTrajectory(
        {'u': noised_data}, domain, cache_id=0,
    )
    epde_search_obj.create_pool(
        data=[trajectory],
        max_deriv_order=(2, 3),
        additional_tokens=[],
    )

    # 3. Параметры оператора — один раз
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator(
        'DeepXDEBasedFitness',
    )
    print(f"[CHECK] operator_params = {operator_params}")

    # 4. Прогоняем каждое уравнение
    results = []
    for label, eq_str in equations:
        print("\n" + "=" * 60)
        print(f"[CHECK] {label}")
        print(f"[CHECK] equation: {eq_str}")
        print("=" * 60)

        try:
            soeq = translate_equation(
                eq_str, epde_search_obj.pool, all_vars=['u'],
            )
            eq = soeq.vals['u']
            eq.main_var_to_explain = 'u'
            eq.weights_internal = np.append(
                np.ones(len(eq.structure) - 1), 0.0,
            )
            eq.weights_internal_evald = True
            eq.weights_final_evald = True

            system = SoEq(epde_search_obj.pool, {})
            system.vals = Chromosome({'u': eq}, {})
            system.moeadd_set = True

            primary = Discrepancy(metric='deepxde', error_metric='rmse')
            instability = Instability()

            fit_operator = SolverBasedFitness(
                param_keys=list(operator_params.keys()),
                objectives=[primary, instability],
                primary=primary,
                backend='deepxde',
                masked=False,
            )
            fit_operator.params = operator_params

            print("[CHECK] Running SolverBasedFitness (DeepXDE)...")
            fit_operator.apply(system, {}, force_out_of_place=False)

            print("\n[CHECK] RESULTS:")
            for var_name, eq_obj in zip(system.vars_to_describe, system.vals):
                fv = getattr(eq_obj, 'fitness_value', None)
                cs = getattr(eq_obj, 'coefficients_stability', None)
                print(f"  var={var_name}")
                print(f"    fitness_value          = {fv}")
                print(f"    coefficients_stability = {cs}")
                print(f"    fitness_calculated     = "
                      f"{getattr(eq_obj, 'fitness_calculated', None)}")
                print(f"    stability_calculated   = "
                      f"{getattr(eq_obj, 'stability_calculated', None)}")

            results.append({
                'label': label,
                'equation': eq_str,
                'fitness': fv,
                'stability': cs,
                'status': 'ok',
            })

        except Exception as exc:
            print(f"[CHECK] FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                'label': label,
                'equation': eq_str,
                'fitness': None,
                'stability': None,
                'status': f'failed: {exc}',
            })

    # 5. Итоговая сводка
    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    for r in results:
        print(f"  {r['label']:<15} "
              f"fitness={r['fitness']}, "
              f"stability={r['stability']}, "
              f"status={r['status']}")

    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    directory = os.path.dirname(os.path.realpath(__file__))
    print(f"[CHECK] folder = {directory}")

    check_equations(directory, noise_level=0.0)