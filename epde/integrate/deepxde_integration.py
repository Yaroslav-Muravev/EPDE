import numpy as np
import os
import sys
from typing import List, Union, Tuple
from scipy.spatial import cKDTree

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var
import deepxde as dde
from abc import ABC, abstractmethod

os.makedirs(os.path.expanduser('~/.deepxde'), exist_ok=True)

class SolverStrategy(ABC):
    @abstractmethod
    def solve(self, eq_list: List[Equation], var_names: List[str],
              grids: List[np.ndarray], data_list: List[np.ndarray],
              adapter: 'DeepXDEAdapter') -> Tuple[List[np.ndarray], float]:
        pass

class Solver1D(SolverStrategy):
    def solve(self, eq_list, var_names, grids, data_list, adapter):
        t_full = grids[0]
        mask = global_var.grid_cache.g_func_mask.flatten()

        t_all = t_full.flatten()[mask]

        # Разделение по времени согласно train_ratio
        unique_t = np.unique(t_all)
        split_idx = int(len(unique_t) * adapter.train_ratio)
        t_train_values = unique_t[:split_idx]
        mask_train = np.isin(t_all, t_train_values)

        t_train = t_all[mask_train]
        data_train_list = [d[mask_train] for d in data_list]

        geom = dde.geometry.TimeDomain(t_all.min(), t_all.max())  # полный временной интервал
        coords_masked = t_train.reshape(-1, 1)

        eps_t = (t_all.max() - t_all.min()) * 1e-5
        initial_idx = np.where(np.abs(t_train - t_all.min()) < eps_t)[0]

        bcs = []
        for var_idx, data_train in enumerate(data_train_list):
            data_masked = data_train.ravel()

            def make_ic_func(indices):
                if len(indices) == 0:
                    return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)
                tree = cKDTree(coords_masked[indices])

                def func(x):
                    if hasattr(x, 'detach'):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = np.asarray(x)
                    _, idx = tree.query(x_np)
                    return data_masked[indices][idx].reshape(-1, 1)
                return func

            if len(initial_idx) > 0:
                bcs.append(dde.icbc.IC(geom, make_ic_func(initial_idx),
                                       lambda _, on_initial: on_initial,
                                       component=var_idx))
            else:
                bcs.append(dde.icbc.IC(geom, make_ic_func([]),
                                       lambda _, on_initial: on_initial,
                                       component=var_idx))

        pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
        data_obj = dde.data.PDE(geom, pde_func, bcs,
                                num_domain=adapter.num_domain,
                                num_boundary=adapter.num_boundary,
                                num_test=adapter.num_test)

        model = adapter._get_or_create_model(data_obj, dim=1, var_count=len(var_names))
        try:
            losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
        except Exception as e:
            print(f"Exception: {e}")
            y_pred = [np.full(data.shape, np.nan) for data in data_list]
            return y_pred, np.nan

        # Предсказание на всех активных точках
        coords_pred = t_full.flatten()[mask].reshape(-1, 1)
        pred = model.predict(coords_pred)
        solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
        return solutions, final_loss
#
# class Solver1D(SolverStrategy):
#     def solve(self, eq_list, var_names, grids, data_list, adapter):
#         t = grids[0]
#         geom = dde.geometry.TimeDomain(t.min(), t.max())
#         mask = global_var.grid_cache.g_func_mask
#         coords_masked = t[mask].reshape(-1, 1)
#
#         eps_t = (t.max() - t.min()) * 1e-5
#         initial_idx = np.where(np.abs(t[mask] - t.min()) < eps_t)[0]
#
#         bcs = []
#         for var_idx, data in enumerate(data_list):
#             data_masked = data.ravel()
#
#             def make_ic_func(indices):
#                 if len(indices) == 0:
#                     return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)  # <-- ИСПРАВЛЕНО
#                 tree = cKDTree(coords_masked[indices])
#
#                 def func(x):
#                     if hasattr(x, 'detach'):
#                         x_np = x.detach().cpu().numpy()
#                     else:
#                         x_np = np.asarray(x)
#                     _, idx = tree.query(x_np)
#                     return data_masked[indices][idx].reshape(-1, 1)
#
#                 return func
#
#             if len(initial_idx) > 0:
#                 bcs.append(dde.icbc.IC(geom, make_ic_func(initial_idx),
#                                        lambda _, on_initial: on_initial,
#                                        component=var_idx))
#             else:
#                 bcs.append(dde.icbc.IC(geom, make_ic_func([]),
#                                        lambda _, on_initial: on_initial,
#                                        component=var_idx))
#
#         pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
#         data_obj = dde.data.PDE(geom, pde_func, bcs,
#                                 num_domain=adapter.num_domain,
#                                 num_boundary=adapter.num_boundary,
#                                 num_test=600)
#
#         model = adapter._get_or_create_model(data_obj, dim=1, var_count=len(var_names))
#         try:
#             losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)
#             final_loss = float(
#                 losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
#         except Exception as e:
#             print(f"Exception: {e}")
#             y_pred = [np.full(data.shape, np.nan) for data in data_list]
#             return y_pred, np.nan
#
#         t_flat = t.reshape(-1, 1)
#         pred = model.predict(t_flat)
#         solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
#         return solutions, final_loss

# class Solver2D(SolverStrategy):
#     def solve(self, eq_list, var_names, grids, data_list, adapter):
#         t, x = grids[0], grids[1]
#         geom = dde.geometry.Interval(x.min(), x.max())
#         timedomain = dde.geometry.TimeDomain(t.min(), t.max())
#         geomtime = dde.geometry.GeometryXTime(geom, timedomain)
#
#         mask = global_var.grid_cache.g_func_mask
#         masked_coords = np.stack([g[mask] for g in grids], axis=1)
#         masked_coords_swapped = masked_coords[:, [1, 0]]
#         eps_x = (x.max() - x.min()) * 1e-5
#         eps_t = (t.max() - t.min()) * 1e-5
#
#         left_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.min()) < eps_x)[0]
#         right_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.max()) < eps_x)[0]
#         initial_idx = np.where(np.abs(masked_coords_swapped[:, 1] - t.min()) < eps_t)[0]
#
#         bcs = []
#         for var_idx, data in enumerate(data_list):
#             data_masked = data.ravel()
#
#             def make_bc_func(indices):
#                 if len(indices) == 0:
#                     return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)  # <- исправлено
#                 tree = cKDTree(masked_coords_swapped[indices])
#
#                 def func(x):
#                     if hasattr(x, 'detach'):
#                         x_np = x.detach().cpu().numpy()
#                     else:
#                         x_np = np.asarray(x)
#                     _, idx = tree.query(x_np)
#                     return data_masked[indices][idx].reshape(-1, 1)
#
#                 return func
#
#             if len(left_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(left_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[0], x.min(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_x),
#                                                 component=var_idx)) # Заменил _.x[0] на _[0]
#             if len(right_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(right_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[0], x.max(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_x),
#                                                 component=var_idx)) # Заменил _.x[0] на _[0]
#             if len(initial_idx) > 0:
#                 bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx),
#                                        lambda _, on_initial: on_initial,
#                                        component=var_idx))
#
#         pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
#         data_obj = dde.data.TimePDE(geomtime, pde_func, bcs,
#                                     num_domain=adapter.num_domain,
#                                     num_boundary=adapter.num_boundary,
#                                     num_initial=adapter.num_initial,
#                                     num_test=500)
#
#         model = adapter._get_or_create_model(data_obj, dim=2, var_count=len(var_names))
#         try:
#             losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)  # <-- ИСПРАВЛЕНО
#             final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
#         except Exception as e:
#             print(f"Exception: {e}")
#             y_pred = [np.full(data.shape, np.nan) for data in data_list]
#             return y_pred, np.nan
#
#         coords_pred = np.stack([x.flatten(), t.flatten()], axis=1)
#         pred = model.predict(coords_pred)
#         solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
#         return solutions, final_loss

class Solver2D(SolverStrategy):
    def solve(self, eq_list, var_names, grids, data_list, adapter):
        t_full, x_full = grids[0], grids[1]
        mask = global_var.grid_cache.g_func_mask.flatten()  # маска активных точек (bool массив)

        # Применяем маску к сетке, получаем только точки с данными
        t_all = t_full.flatten()[mask]
        x_all = x_full.flatten()[mask]

        # Теперь t_all и x_all имеют длину = data_list[0].shape[0] (4428)
        unique_t = np.unique(t_all)
        half_idx = len(unique_t) // 2
        t_train_values = unique_t[:half_idx]
        mask_flat = np.isin(t_all, t_train_values)  # маска для train-точек (длина 4428)

        # Данные уже плоские и соответствуют маске, поэтому можно безопасно применять mask_flat
        t_train = t_all[mask_flat]
        x_train = x_all[mask_flat]
        data_train_list = [d[mask_flat] for d in data_list]

        # Геометрия и граничные условия строятся на основе тренировочных точек,
        # но можно оставить полные пределы (x_full.min()/max() и t_all.min()/max())
        geom = dde.geometry.Interval(x_full.min(), x_full.max())
        timedomain = dde.geometry.TimeDomain(t_all.min(), t_all.max())
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        # Координаты для KDTree (x, t) — только тренировочные
        masked_coords_train = np.stack([t_train, x_train], axis=1)  # (t, x)
        masked_coords_swapped_train = masked_coords_train[:, [1, 0]]  # (x, t)

        eps_x = (x_full.max() - x_full.min()) * 1e-5
        eps_t = (t_all.max() - t_all.min()) * 1e-5

        left_idx = np.where(np.abs(masked_coords_swapped_train[:, 0] - x_full.min()) < eps_x)[0]
        right_idx = np.where(np.abs(masked_coords_swapped_train[:, 0] - x_full.max()) < eps_x)[0]
        initial_idx = np.where(np.abs(masked_coords_swapped_train[:, 1] - t_all.min()) < eps_t)[0]

        bcs = []
        for var_idx, data_train in enumerate(data_train_list):
            data_masked = data_train.ravel()

            def make_bc_func(indices):
                if len(indices) == 0:
                    return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)
                tree = cKDTree(masked_coords_swapped_train[indices])
                def func(x):
                    if hasattr(x, 'detach'):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = np.asarray(x)
                    _, idx = tree.query(x_np)
                    return data_masked[indices][idx].reshape(-1, 1)
                return func

            if len(left_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(left_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[0], x_full.min(), rtol=1e-5, atol=eps_x),
                            component=var_idx))
            if len(right_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(right_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[0], x_full.max(), rtol=1e-5, atol=eps_x),
                            component=var_idx))
            if len(initial_idx) > 0:
                bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx),
                            lambda _, on_initial: on_initial,
                            component=var_idx))

        pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
        data_obj = dde.data.TimePDE(geomtime, pde_func, bcs,
                                    num_domain=adapter.num_domain,
                                    num_boundary=adapter.num_boundary,
                                    num_initial=adapter.num_initial,
                                    num_test=504)

        model = adapter._get_or_create_model(data_obj, dim=2, var_count=len(var_names))
        try:
            losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
        except Exception as e:
            print(f"Exception: {e}")
            y_pred = [np.full(d.shape, np.nan) for d in data_list]
            return y_pred, np.nan

        # Предсказание на полной сетке
        mask_active = global_var.grid_cache.g_func_mask.flatten()
        x_active = x_full.flatten()[mask_active]
        t_active = t_full.flatten()[mask_active]
        coords_pred = np.stack([x_active, t_active], axis=1)
        pred = model.predict(coords_pred)
        solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
        return solutions, final_loss


class Solver3D(SolverStrategy):
    def solve(self, eq_list, var_names, grids, data_list, adapter):
        t_full, x_full, y_full = grids[0], grids[1], grids[2]
        mask = global_var.grid_cache.g_func_mask.flatten()

        t_all = t_full.flatten()[mask]
        x_all = x_full.flatten()[mask]
        y_all = y_full.flatten()[mask]

        # Разделение по времени согласно train_ratio
        unique_t = np.unique(t_all)
        split_idx = int(len(unique_t) * adapter.train_ratio)
        t_train_values = unique_t[:split_idx]
        mask_train = np.isin(t_all, t_train_values)

        t_train = t_all[mask_train]
        x_train = x_all[mask_train]
        y_train = y_all[mask_train]
        data_train_list = [d[mask_train] for d in data_list]

        # Геометрия (полные пределы)
        geom = dde.geometry.Rectangle([x_full.min(), y_full.min()],
                                      [x_full.max(), y_full.max()])
        timedomain = dde.geometry.TimeDomain(t_all.min(), t_all.max())
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        # Координаты для KDTree: (x, y, t) – порядок, используемый в оригинале
        masked_coords_train = np.stack([x_train, y_train, t_train], axis=1)
        eps_x = (x_full.max() - x_full.min()) * 1e-5
        eps_y = (y_full.max() - y_full.min()) * 1e-5
        eps_t = (t_all.max() - t_all.min()) * 1e-5

        x_min_idx = np.where(np.abs(masked_coords_train[:, 0] - x_full.min()) < eps_x)[0]
        x_max_idx = np.where(np.abs(masked_coords_train[:, 0] - x_full.max()) < eps_x)[0]
        y_min_idx = np.where(np.abs(masked_coords_train[:, 1] - y_full.min()) < eps_y)[0]
        y_max_idx = np.where(np.abs(masked_coords_train[:, 1] - y_full.max()) < eps_y)[0]
        initial_idx = np.where(np.abs(masked_coords_train[:, 2] - t_all.min()) < eps_t)[0]

        bcs = []
        for var_idx, data_train in enumerate(data_train_list):
            data_masked = data_train.ravel()

            def make_bc_func(indices):
                if len(indices) == 0:
                    return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)
                tree = cKDTree(masked_coords_train[indices])

                def func(x):
                    if hasattr(x, 'detach'):
                        x_np = x.detach().cpu().numpy()
                    else:
                        x_np = np.asarray(x)
                    _, idx = tree.query(x_np)
                    return data_masked[indices][idx].reshape(-1, 1)
                return func

            if len(x_min_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(x_min_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[0], x_full.min(), rtol=1e-5, atol=eps_x),
                            component=var_idx))
            if len(x_max_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(x_max_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[0], x_full.max(), rtol=1e-5, atol=eps_x),
                            component=var_idx))
            if len(y_min_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(y_min_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[1], y_full.min(), rtol=1e-5, atol=eps_y),
                            component=var_idx))
            if len(y_max_idx) > 0:
                bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(y_max_idx),
                            lambda _, on_boundary: on_boundary and np.isclose(_[1], y_full.max(), rtol=1e-5, atol=eps_y),
                            component=var_idx))
            if len(initial_idx) > 0:
                bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx),
                            lambda _, on_initial: on_initial,
                            component=var_idx))

        pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
        data_obj = dde.data.TimePDE(geomtime, pde_func, bcs,
                                    num_domain=adapter.num_domain,
                                    num_boundary=adapter.num_boundary,
                                    num_initial=adapter.num_initial,
                                    num_test=adapter.num_test)

        model = adapter._get_or_create_model(data_obj, dim=3, var_count=len(var_names))
        try:
            losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
        except Exception as e:
            print(f"Exception: {e}")
            y_pred = [np.full(data.shape, np.nan) for data in data_list]
            return y_pred, np.nan

        # Предсказание на всех активных точках (порядок x, y, t)
        coords_pred = np.stack([x_all, y_all, t_all], axis=1)
        pred = model.predict(coords_pred)
        solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
        return solutions, final_loss

# class Solver3D(SolverStrategy):
#     def solve(self, eq_list, var_names, grids, data_list, adapter):
#         t, x, y = grids[0], grids[1], grids[2]
#         geom = dde.geometry.Rectangle([x.min(), y.min()], [x.max(), y.max()])
#         timedomain = dde.geometry.TimeDomain(t.min(), t.max())
#         geomtime = dde.geometry.GeometryXTime(geom, timedomain)
#
#         mask = global_var.grid_cache.g_func_mask
#         masked_coords = np.stack([g[mask] for g in grids], axis=1)
#         masked_coords_swapped = masked_coords[:, [1, 2, 0]]
#         eps_x = (x.max() - x.min()) * 1e-5
#         eps_y = (y.max() - y.min()) * 1e-5
#         eps_t = (t.max() - t.min()) * 1e-5
#
#         x_min_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.min()) < eps_x)[0]
#         x_max_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.max()) < eps_x)[0]
#         y_min_idx = np.where(np.abs(masked_coords_swapped[:, 1] - y.min()) < eps_y)[0]
#         y_max_idx = np.where(np.abs(masked_coords_swapped[:, 1] - y.max()) < eps_y)[0]
#         initial_idx = np.where(np.abs(masked_coords_swapped[:, 2] - t.min()) < eps_t)[0]
#
#         bcs = []
#         for var_idx, data in enumerate(data_list):
#             data_masked = data.ravel()
#
#             def make_bc_func(indices):
#                 if len(indices) == 0:
#                     return lambda x: np.full((x.shape[0], 1), adapter.fallback_bc_value)  # <- исправлено
#                 tree = cKDTree(masked_coords_swapped[indices])
#
#                 def func(x):
#                     if hasattr(x, 'detach'):
#                         x_np = x.detach().cpu().numpy()
#                     else:
#                         x_np = np.asarray(x)
#                     _, idx = tree.query(x_np)
#                     return data_masked[indices][idx].reshape(-1, 1)
#
#                 return func
#
#             if len(x_min_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(x_min_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[0], x.min(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_x),
#                                                 component=var_idx)) # Заменил _.x[0] на _[0]
#             if len(x_max_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(x_max_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[0], x.max(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_x),
#                                                 component=var_idx)) # Заменил _.x[0] на _[0]
#             if len(y_min_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(y_min_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[1], y.min(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_y),
#                                                 component=var_idx)) # Заменил _.x[1] на _[1]
#             if len(y_max_idx) > 0:
#                 bcs.append(dde.icbc.DirichletBC(geomtime, make_bc_func(y_max_idx),
#                                                 lambda _, on_boundary: on_boundary and np.isclose(_[1], y.max(),
#                                                                                                   rtol=1e-5,
#                                                                                                   atol=eps_y),
#                                                 component=var_idx)) # Заменил _.x[1] на _[1]
#             if len(initial_idx) > 0:
#                 bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx),
#                                        lambda _, on_initial: on_initial,
#                                        component=var_idx))
#
#         pde_func = adapter._equation_system_to_pde_func(dde, eq_list, var_names)
#         data_obj = dde.data.TimePDE(geomtime, pde_func, bcs,
#                                     num_domain=adapter.num_domain,
#                                     num_boundary=adapter.num_boundary,
#                                     num_initial=adapter.num_initial,
#                                     num_test=500)
#
#         model = adapter._get_or_create_model(data_obj, dim=3, var_count=len(var_names))
#         try:
#             losshistory, train_state = model.train(iterations=adapter.iterations, verbose=adapter.verbose)  # <-- ИСПРАВЛЕНО
#             final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else np.nan
#         except Exception as e:
#             print(f"Exception: {e}")
#             y_pred = [np.full(data.shape, np.nan) for data in data_list]
#             return y_pred, np.nan
#
#         coords_pred = np.stack([x.flatten(), y.flatten(), t.flatten()], axis=1)
#         pred = model.predict(coords_pred)
#         solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
#         return solutions, final_loss


class DeepXDEAdapter:
    def __init__(self, pretrained_net=None, **config):
        self.pretrained_net = pretrained_net
        self.config = config or {}
        self.net = self.config.get('net', [50, 50, 50, 50])
        self.activation = self.config.get('activation', 'tanh')
        self.optimizer = self.config.get('optimizer', 'adam')
        self.lr = self.config.get('lr', 1e-3)
        self.kernel_initializer = self.config.get('kernel_initializer', 'Glorot normal')
        self.num_domain = int(self.config.get('num_domain', 2000))
        self.num_boundary = int(self.config.get('num_boundary', 500))
        self.num_initial = int(self.config.get('num_initial', 500))
        #self.epochs = int(self.config.get('epochs', 10000))
        self.iterations = int(self.config.get('iterations', 1000))
        self.bc_type = self.config.get('bc_type', 'Dirichlet')
        self.fallback_bc_value = self.config.get('fallback_bc_value', 0.0)
        self.verbose = config.get('verbose', False)
        self.train_ratio = float(config.get('train_ratio', 1.0))
        self.num_test = int(self.config.get('num_test', 500))

        self.coordinate_mapping = self.config.get('coordinate_mapping', None)
        self.coord_names = None
        self.coord_map = None

        self._solvers = {
            1: Solver1D(),
            2: Solver2D(),
            3: Solver3D(),
        }

        self._model = None

    def _get_or_create_model(self, data_obj, dim, var_count):
        if self._model is None:
            layer_size = [dim] + self.net + [var_count]
            net = dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
            model = dde.Model(data_obj, net)
            model.compile(self.optimizer, lr=self.lr, verbose=self.verbose)
            self._model = model
        else:
            def reset_weights(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self._model.net.apply(reset_weights)
            self._model.data = data_obj
        return self._model

    def _set_coordinate_info(self, coord_names):
        self.coord_names = coord_names
        if self.coordinate_mapping is not None:
            self.coord_map = self.coordinate_mapping
        else:
            spatial_dim = len(coord_names) - 1
            self.coord_map = {}
            for i, name in enumerate(coord_names):
                if i == 0:
                    self.coord_map[name] = spatial_dim
                else:
                    self.coord_map[name] = i - 1

    def _equation_system_to_pde_func(self, dde, eq_list, var_names):
        var_idx_map = {name: i for i, name in enumerate(var_names)}

        def pde(x, y):
            # Считаем невязку
            residuals = []
            for eq_idx, eq in enumerate(eq_list):
                use_weights = getattr(eq, "weights_final_evald", False) and hasattr(eq, "weights_final")
                residual = y[:, eq_idx:eq_idx + 1] * 0.0
                all_terms = eq.structure
                tgt = eq.target_idx
                for term_idx, term in enumerate(all_terms):
                    if term_idx == tgt:
                        continue
                    if use_weights and len(eq.weights_final) > term_idx:
                        coeff = float(eq.weights_final[term_idx])
                    else:
                        coeff = 1.0
                    term_val = 1.0
                    for factor in term.structure:
                        fv = self._factor_value_with_map(dde, factor, x, y, self.coord_map, var_idx_map)
                        term_val *= fv
                    residual += coeff * term_val
                if use_weights and len(eq.weights_final) > len(all_terms):
                    residual += float(eq.weights_final[-1]) * (y[:, 0:1] * 0.0 + 1.0)
                target = eq.target
                if target is not None:
                    target_val = 1.0
                    for factor in target.structure:
                        fv = self._factor_value_with_map(dde, factor, x, y, self.coord_map, var_idx_map)
                        target_val *= fv
                    residual -= target_val
                residuals.append(residual)
            return residuals

        return pde

    def _factor_value_with_map(self, dde, factor, x, y, coord_map, var_idx_map=None):
        # Derivative
        if getattr(factor, "is_deriv", False) and getattr(factor, "deriv_code", None):
            var_name = getattr(factor, "variable", None)
            if var_name is None:
                return y[:, 0:1] * 0.0
            idx = var_idx_map.get(var_name, 0) if var_idx_map is not None else 0
            val = y[:, idx:idx + 1]
            for ax in factor.deriv_code:
                if ax is None:
                    continue
                try:
                    ax_int = int(ax)
                except (ValueError, TypeError):
                    continue
                coord_name = self.coord_names[ax_int]
                dde_ax = coord_map.get(coord_name, None)
                if dde_ax is not None:
                    val = dde.grad.jacobian(val, x, i=0, j=dde_ax)
                else:
                    return y[:, 0:1] * 0.0
            return val

        # Main variable u (or other)
        if getattr(factor, "variable", None) is not None:
            var_name = factor.variable
            idx = var_idx_map.get(var_name, 0) if var_idx_map is not None else 0
            params = getattr(factor, "params", [1.0])
            p = float(params[-1])
            return y[:, idx:idx + 1] ** p

        # Constant
        if len(getattr(factor, "structure", [])) == 0 or 'const' in str(getattr(factor, "name", "")).lower():
            return y[:, 0:1] * 0.0 + 1.0

        # Grid token (t, x, y...)
        label = getattr(factor, "cache_label", None)
        if label:
            if isinstance(label, tuple):
                label = str(label[0]).lower()
            else:
                label = str(label).lower()
            idx = coord_map.get(label, None)
            if idx is not None:
                return x[:, int(idx):int(idx) + 1]
        return y[:, 0:1] * 0.0 + 1.0

    def solve(self, equation_or_system, grids: list, data):
        dim = len(grids)
        solver = self._solvers.get(dim)

        keys, _ = global_var.grid_cache.get_all(mode='numpy')
        self._set_coordinate_info(keys)

        if isinstance(equation_or_system, Equation):
            eq_list = [equation_or_system]
            var_names = [equation_or_system.main_var_to_explain]
            if isinstance(data, np.ndarray):
                data_list = [data]
            else:
                data_list = data
        elif isinstance(equation_or_system, SoEq):
            var_names = equation_or_system.vars_to_describe
            eq_list = [equation_or_system.vals[var] for var in equation_or_system.vars_to_describe]
            if isinstance(data, np.ndarray):
                raise ValueError("For SoEq, data must be a list of arrays (one per variable).")
            data_list = data
        else:
            raise TypeError("Unsupported equation type")

        return solver.solve(eq_list, var_names, grids, data_list, self)