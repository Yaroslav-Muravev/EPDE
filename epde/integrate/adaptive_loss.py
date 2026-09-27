# epde/integrate/adaptive_loss.py
import os
import numpy as np
import deepxde as dde

os.makedirs(os.path.expanduser('~/.deepxde'), exist_ok=True)


# =========================================================================
# Оценки масштабов (variance yardsticks)
# =========================================================================

def compute_variance_scale(data_arrays, t_grid, t_split_frac=0.5):
    """Одномерный случай (только t).

    Возвращает [Var(u_t), Var(u), Var(u), Var(u)] — PDE + 3 BC-слота
    (Dirichlet/Neumann/IC/Data — все BC-подобные оцениваются одинаково).
    """
    t_thr = t_grid.min() + t_split_frac * (t_grid.max() - t_grid.min())
    mask_t = t_grid <= t_thr

    u = data_arrays[0]
    u_train = u[mask_t]
    t_train = t_grid[mask_t]

    if t_train.size > 1:
        order = np.argsort(t_train)
        u_t = np.gradient(u_train[order], t_train[order])
    else:
        u_t = np.zeros_like(u_train)

    var_phys = float(np.var(u_t) + 1e-8)
    var_data = float(np.var(u_train) + 1e-8)
    return [var_phys, var_data, var_data, var_data]


def _reshape_grid(u_data, t_all, x_all):
    """Пытается восстановить (n_t, n_x) форму из плоских массивов.

    Возвращает (u2d, t_axis) или (None, None), если маска не прямоугольная.
    """
    unique_t = np.unique(t_all)
    unique_x = np.unique(x_all)
    n_t, n_x = len(unique_t), len(unique_x)
    if u_data.size != n_t * n_x:
        return None, None
    return u_data.reshape(n_t, n_x), unique_t


def compute_variance_scale_2d(u_data, t_all, x_all, t_split_frac=0.5,
                              rhs_func=None):
    """2D (t + x).

    Var_pde оценивается как max(Var(u_t_FD), Var(F(u))), где F — правая
    часть уравнения (если rhs_func передан). На корректном уравнении
    F(u) ≈ u_t, и max защищает от недооценки из-за FD-шума.

    Все BC-подобные слоты (Dirichlet, IC, PointSetBC) получают одну и ту
    же Var(u_data): это соответствует физике — данные и граничные
    значения измеряются в одних единицах.
    """
    u_data = np.asarray(u_data).reshape(-1)
    u2d, t_axis = _reshape_grid(u_data, t_all, x_all)

    if u2d is None:
        # Нерегулярная маска: единая грубая оценка
        var_u = float(np.var(u_data) + 1e-8)
        print(f"[var_scale_2d] non-rectangular mask; "
              f"fallback Var(u)={var_u:.4g}")
        return [var_u, var_u, var_u, var_u]

    n_t = u2d.shape[0]
    n_train = max(3, int(np.sum(
        t_axis <= t_axis.min() + t_split_frac * (t_axis.max() - t_axis.min())
    )))
    n_train = min(n_train, n_t)

    u_train = u2d[:n_train, :]

    # Var(u_t): центральная разность по времени
    dt = np.diff(t_axis[:n_train])
    du = np.diff(u_train, axis=0)
    u_t = du / dt[:, None] if dt.size else np.zeros_like(u_train)
    var_u_t = float(np.var(u_t) + 1e-8)

    # Var(u) — для BC/IC/Data
    var_u = float(np.var(u_train) + 1e-8)

    # Var(F(u)) — для PDE, если есть RHS
    var_F = None
    if rhs_func is not None:
        try:
            F_u = rhs_func(u_train, t_axis[:n_train])
            var_F = float(np.var(F_u) + 1e-8)
        except Exception as e:
            print(f"[var_scale_2d] rhs_func failed: {e}; "
                  f"using Var(u_t) only")

    var_pde = var_u_t if var_F is None else max(var_u_t, var_F)

    print(f"[var_scale_2d] Var(u_t)={var_u_t:.4g}, "
          f"Var(F)={var_F if var_F is not None else 'n/a'}, "
          f"Var(u)={var_u:.4g}  ->  Var_pde={var_pde:.4g}")

    return [var_pde, var_u, var_u, var_u]


def compute_variance_scale_3d(u_data, t_all, x_all, y_all,
                              t_split_frac=0.5, rhs_func=None):
    """3D (t + x + y). Аналог 2D-варианта.

    Если сетка регулярная — считает Var(u_t) через центральную разность.
    Иначе падает на общую Var(u).
    """
    u_data = np.asarray(u_data).reshape(-1)
    unique_t = np.unique(t_all)
    unique_x = np.unique(x_all)
    unique_y = np.unique(y_all)
    n_t, n_x, n_y = len(unique_t), len(unique_x), len(unique_y)

    if u_data.size != n_t * n_x * n_y:
        var_u = float(np.var(u_data) + 1e-8)
        print(f"[var_scale_3d] non-rectangular mask; "
              f"fallback Var(u)={var_u:.4g}")
        return [var_u, var_u, var_u, var_u]

    u = u_data.reshape(n_t, n_x, n_y)
    n_train = max(3, int(np.sum(
        unique_t <= unique_t.min()
        + t_split_frac * (unique_t.max() - unique_t.min())
    )))
    n_train = min(n_train, n_t)
    u_train = u[:n_train]

    dt = np.diff(unique_t[:n_train])
    du = np.diff(u_train, axis=0)
    u_t = du / dt[:, None, None] if dt.size else np.zeros_like(u_train)

    var_u_t = float(np.var(u_t) + 1e-8)
    var_u = float(np.var(u_train) + 1e-8)

    var_F = None
    if rhs_func is not None:
        try:
            F_u = rhs_func(u_train, unique_t[:n_train])
            var_F = float(np.var(F_u) + 1e-8)
        except Exception as e:
            print(f"[var_scale_3d] rhs_func failed: {e}")

    var_pde = var_u_t if var_F is None else max(var_u_t, var_F)
    print(f"[var_scale_3d] Var(u_t)={var_u_t:.4g}, "
          f"Var(F)={var_F if var_F is not None else 'n/a'}, "
          f"Var(u)={var_u:.4g}  ->  Var_pde={var_pde:.4g}")

    return [var_pde, var_u, var_u, var_u]


# =========================================================================
# Callback: фиксированные веса лосса, нормированные на Var таргета
# =========================================================================

class AdaptiveLoss(dde.callbacks.Callback):
    """Фиксирует loss_weights = 1 / Var(target) один раз перед обучением.

    Зачем. В одном лоссе складываются величины разных размерностей:
    PDE-невязка (единицы u/t), BC/IC/Data (единицы u). Без нормировки
    их относительный вклад определяется масштабом задачи, и баланс
    теряет смысл при смене единиц. Деление каждой компоненты на
    Var её таргета делает все слагаемые безразмерными и ~O(1).

    Parameters
    ----------
    model : dde.Model
        Модель, которая обучается; перекомпилируется с новыми весами.
    optimizer : str, optional
        Имя оптимизатора (``'adam'``). Обязательно передать: у
        ``dde.Model`` нет публичного атрибута ``optimizer``.
    lr : float, optional
        Learning rate. Обязательно передать по той же причине.
    variance_scale : list[float], optional
        Вектор [Var_pde, Var_bc, Var_ic, Var_data, ...]. Если длиннее,
        чем число loss-компонент модели — обрезается; если короче —
        добивается последним элементом.
    epsilon : float
        Мелкая добавка при делении.
    weight_min, weight_max : float, optional
        Клиппирование весов до нормализации.
    normalize : bool
        Нормировать ли веса так, чтобы их сумма была равна 1.
        ``True`` удобно для интерпретации («доля лосса»), но меняет
        эффективный learning rate. ``False`` даёт чистую нормировку
        на дисперсию.
    """

    def __init__(self, model,
                 optimizer=None, lr=None,
                 variance_scale=None,
                 epsilon=1e-8,
                 weight_min=None, weight_max=None,
                 normalize=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.epsilon = float(epsilon)
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.normalize = bool(normalize)

        if variance_scale is not None:
            self.variance_scale = np.atleast_1d(
                np.asarray(variance_scale, dtype=float))
        else:
            self.variance_scale = None

        self.weights = None

    # ---------- вспомогательное ----------

    def _n_model_losses(self):
        """Сколько loss-компонент DeepXDE реально создал для модели.

        DeepXDE формирует список ``model.data.losses`` (строковые
        идентификаторы) в момент построения. Если его нет — считаем
        как ``1 (PDE) + len(bcs)``.
        """
        data = getattr(self.model, 'data', None)
        if data is None:
            return None
        losses = getattr(data, 'losses', None)
        if losses is not None:
            try:
                return len(losses)
            except TypeError:
                pass
        bcs = getattr(data, 'bcs', None)
        if bcs is not None:
            return 1 + len(bcs)
        return None

    def _clip(self, w):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        if self.weight_min is not None:
            w = np.maximum(w, self.weight_min)
        if self.weight_max is not None:
            w = np.minimum(w, self.weight_max)
        return w

    def _normalize_to_sum_one(self, w):
        s = float(np.sum(w))
        if s <= 0:
            return np.ones_like(w) / w.size
        return w / s

    def _apply_weights(self, new_weights):
        self.weights = np.atleast_1d(np.asarray(new_weights, dtype=float))
        try:
            self.model.compile(
                self.optimizer,           # передан из SolverND.solve
                lr=self.lr,               # передан из SolverND.solve
                loss_weights=self.weights.tolist(),
            )
            print(f"[AdaptiveLoss] Recompiled with "
                  f"{len(self.weights)} weights "
                  f"(optimizer={self.optimizer}, lr={self.lr}).")
        except Exception as e:
            print(f"[AdaptiveLoss] Recompile failed: {e}")

    # ---------- callbacks ----------

    def on_train_begin(self):
        if self.variance_scale is None:
            print("[AdaptiveLoss] variance_scale is None — "
                  "keeping default weights.")
            return

        vscale = np.atleast_1d(np.asarray(self.variance_scale, dtype=float))
        n = self._n_model_losses()
        print(f"[AdaptiveLoss] model declares {n} losses; "
              f"variance_scale has {len(vscale)} entries")

        if n is not None and len(vscale) != n:
            if len(vscale) > n:
                vscale = vscale[:n]
                print(f"[AdaptiveLoss] variance_scale truncated to {n}")
            else:
                fill = vscale[-1]
                vscale = np.concatenate(
                    [vscale, np.full(n - len(vscale), fill)])
                print(f"[AdaptiveLoss] variance_scale padded to {n} "
                      f"with {fill:.4g}")
            self.variance_scale = vscale

        w = 1.0 / (vscale + self.epsilon)
        w = self._clip(w)
        if self.normalize:
            w = self._normalize_to_sum_one(w)
        self._apply_weights(w)

        print(f"[AdaptiveLoss] Fixed weights: "
              f"{np.round(self.weights, 6).tolist()}")
        print(f"[AdaptiveLoss] variance_scale: "
              f"{self.variance_scale.tolist()}")

    def on_epoch_end(self):
        # Веса фиксированы — динамики нет.
        pass