import deepxde as dde
import numpy as np


def compute_variance_scale(data_arrays, t_grid, t_split_frac=0.8):
    """
    data_arrays: list of 1D np.ndarray (inner-domain), по одной на переменную
    t_grid: 1D np.ndarray (inner-domain, тот же порядок, что и data_arrays)
    t_split_frac: доля времени, которая считается тренировочной

    Возвращает вектор: [Var(u_t), Var(u), Var(u), ...] (PDE, BC, IC)
    """
    t_thr = t_grid.min() + t_split_frac * (t_grid.max() - t_grid.min())
    mask_t = t_grid <= t_thr

    u = data_arrays[0]
    u_train = u[mask_t]
    t_train = t_grid[mask_t]

    u_t = np.gradient(u_train, t_train) if t_train.size > 1 else np.zeros_like(u_train)

    var_phys = float(np.var(u_t) + 1e-8)
    var_data = float(np.var(u_train) + 1e-8)

    return [var_phys, var_data, var_data, var_data]


def compute_variance_scale_2d(u_data, t_all, x_all, t_split_frac=0.8):
    """
    Вариант для 2D (t + x). u_data, t_all, x_all — 1D inner-domain массивы.
    Восстанавливает форму (n_t, n_x), считает u_t центральной разностью,
    возвращает [Var(u_t), Var(U), Var(U), Var(U)].
    """
    unique_t = np.unique(t_all)
    unique_x = np.unique(x_all)
    n_t = len(unique_t)
    n_x = len(unique_x)

    # Проверка: суммарный размер должен совпадать
    assert u_data.size == n_t * n_x, (
        f"u_data size {u_data.size} != n_t*n_x = {n_t}*{n_x}")

    # Восстанавливаем форму. Предполагаем C-развёртку (t — медленная ось).
    u = u_data.reshape(n_t, n_x)
    t_axis = unique_t  # 1D вдоль оси 0

    # Тренировочная часть по времени
    t_thr = unique_t.min() + t_split_frac * (unique_t.max() - unique_t.min())
    n_train = np.sum(unique_t <= t_thr)
    if n_train < 3:
        n_train = min(3, n_t)
    u_train = u[:n_train, :]

    # u_t: центральная разность вдоль оси времени
    dt = np.diff(t_axis[:n_train])   # длины (n_train - 1,)
    du = np.diff(u_train, axis=0)    # (n_train - 1, n_x)
    u_t = du / dt[:, None]           # (n_train - 1, n_x) — корректный broadcast

    var_phys = float(np.var(u_t) + 1e-8)
    var_data = float(np.var(u_train) + 1e-8)

    return [var_phys, var_data, var_data, var_data]


class AdaptiveLoss(dde.callbacks.Callback):
    """
    Callback для установки ФИКСИРОВАННЫХ весов лосса на основе дисперсий
    компонент (идея научного руководителя).

    Логика:
      1. Один раз перед обучением задаются yardsticks:
         Var(u_t_FD) — дисперсия производной по времени (PDE-компонент),
         Var(U)      — дисперсия данных (BC / IC / Data).
      2. Веса = 1 / Var, нормированные так, чтобы сумма = 1.
      3. Больше веса НЕ меняются в процессе обучения.

    Параметры
    ----------
    variance_scale : list[float], optional
        Готовый вектор [Var_pde, Var_bc, Var_ic, ...]. Если None — веса
        остаются дефолтными, и callback ничего не делает.
    weight_min, weight_max : float, optional
        Границы для весов до нормализации (защита от крайних значений).
    """

    def __init__(self, model,
                 optimizer=None, lr=None,
                 variance_scale=None,
                 epsilon=1e-8,
                 weight_min=None, weight_max=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.epsilon = epsilon
        self.weight_min = weight_min
        self.weight_max = weight_max

        if variance_scale is not None:
            self.variance_scale = np.atleast_1d(
                np.asarray(variance_scale, dtype=float))
        else:
            self.variance_scale = None

        self.weights = None

    def on_train_begin(self):
        if self.variance_scale is None:
            print("[AdaptiveLoss] variance_scale is None — "
                  "keeping default weights.")
            return
        # Веса = 1 / Var, нормированные
        w = 1.0 / (self.variance_scale + self.epsilon)
        w = self._clip(w)
        self._apply_weights(w)
        print(f"[AdaptiveLoss] Fixed weights: "
              f"{np.round(self.weights, 6).tolist()}")
        print(f"[AdaptiveLoss] variance_scale: "
              f"{self.variance_scale.tolist()}")

    def on_epoch_end(self):
        # Веса фиксированы — динамики нет.
        pass

    def _clip(self, w):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        if self.weight_min is not None:
            w = np.maximum(w, self.weight_min)
        if self.weight_max is not None:
            w = np.minimum(w, self.weight_max)
        return self._normalize(w)

    def _normalize(self, w):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        s = np.sum(w)
        if s <= 0:
            return np.ones_like(w) / w.size
        return w / s

    def _apply_weights(self, new_weights):
        self.weights = new_weights
        try:
            self.model.compile(
                self.optimizer or self.model.optimizer,
                lr=self.lr or self.model.lr,
                loss_weights=self.weights.tolist(),
            )
        except Exception as e:
            print(f"[AdaptiveLoss] Recompile failed: {e}")