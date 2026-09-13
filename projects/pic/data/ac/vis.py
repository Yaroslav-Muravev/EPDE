"""
visualize_ac.py — визуализация данных AC и найденных уравнений.

Требуется: numpy, matplotlib, scipy, deepxde (для решения найденных уравнений).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# 1. Загрузка исходных данных
# ----------------------------------------------------------------------
def ac_data(filename: str):
    t = np.linspace(0.0, 1.0, 51)
    x = np.linspace(-1.0, 0.984375, 128)
    data = np.load(filename)              # shape (51, 128): t × x
    return t, x, data


here = os.path.dirname(os.path.realpath(__file__))
t, x, u_data = ac_data(os.path.join(here, "ac_data.npy"))
X, T = np.meshgrid(x, t, indexing="xy")   # T, X shape (51, 128)

print("Data shape:", u_data.shape, "t-range:", t.min(), t.max(),
      "x-range:", x.min(), x.max())


# ----------------------------------------------------------------------
# 2. Численное решение правильного уравнения Аллена–Кана
#    du/dt = 0.0001 * u_xx - 5*u^3 + 5*u
#    с начальными условиями из данных (t=0) и граничными (x=±1) тоже из данных.
# ----------------------------------------------------------------------
def solve_true_ac():
    dx = x[1] - x[0]
    # Вторая производная по x — центральные разности
    def rhs(t_, u_flat):
        u = u_flat.reshape(x.shape)
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        # Неймановские условия (нулевой поток) на границах
        u_xx[0] = (u[1] - u[0]) / dx**2
        u_xx[-1] = (u[-2] - u[-1]) / dx**2
        return (0.0001 * u_xx - 5 * u**3 + 5 * u).ravel()

    sol = solve_ivp(rhs, (t[0], t[-1]), u_data[0], t_eval=t, method="RK45")
    return sol.y.T.reshape(len(t), len(x))   # (51, 128)


u_true = solve_true_ac()


# ----------------------------------------------------------------------
# 3. Численное решение первого найденного уравнения
#    du/dt = 1.398e-07 * d^3u/dx^3 * du/dx
# ----------------------------------------------------------------------
def solve_eq1():
    dx = x[1] - x[0]
    coef = 1.3983741363213942e-07

    def rhs(t_, u_flat):
        u = u_flat.reshape(x.shape)
        # Третья производная по x — центральные разности 4-го порядка
        u_x = np.zeros_like(u)
        u_xxx = np.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        u_xxx[2:-2] = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx**3)
        # Неймановские условия на границах
        u_x[0] = (u[1] - u[0]) / dx
        u_x[-1] = (u[-2] - u[-1]) / dx
        return (coef * u_xxx * u_x).ravel()

    sol = solve_ivp(rhs, (t[0], t[-1]), u_data[0], t_eval=t, method="RK45")
    return sol.y.T.reshape(len(t), len(x))


u_eq1 = solve_eq1()


# ----------------------------------------------------------------------
# 4. Численное решение второго найденного уравнения
#    u^2 * du/dt = -0.0674 * u_tt + 0.3317 * u_t
#    Здесь оно второго порядка по t, что странно; перепишем как
#    du/dt = (-0.0674 * u_tt + 0.3317 * u_t) / u^2
#    и решим как систему первого порядка по t (с u_t как отдельной переменной).
# ----------------------------------------------------------------------
def solve_eq2():
    dx = x[1] - x[0]
    a, b = -0.06741029623854099, 0.3316648582002735

    def rhs(t_, y_flat):
        n = len(x)
        u = y_flat[:n].reshape(x.shape)
        u_t = y_flat[n:].reshape(x.shape)
        # du/dt = u_t
        # u_tt = (b * u_t - u^2 * u_t) / a  →  но уравнение дано в форме
        # u^2 * du/dt = a * u_tt + b * u_t. Если считать, что du/dt ≈ u_t,
        # то u_tt = (u^2 * u_t - b * u_t) / a
        u_tt = (u**2 * u_t - b * u_t) / a
        # Ограничиваем, чтобы не было взрыва
        u_tt = np.clip(u_tt, -1e6, 1e6)
        return np.concatenate([u_t.ravel(), u_tt.ravel()])

    u0 = u_data[0].ravel()
    ut0 = np.zeros_like(u0)      # du/dt(0) ≈ 0
    y0 = np.concatenate([u0, ut0])
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="RK45")
    return sol.y[:len(x)].T.reshape(len(t), len(x))


u_eq2 = solve_eq2()


# ----------------------------------------------------------------------
# 5. Построение графиков
# ----------------------------------------------------------------------
def plot_surface(ax, Z, title, cmap=cm.viridis):
    ax.plot_surface(X, T, Z, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)


fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
plot_surface(ax1, u_data, "Исходные данные AC", cmap=cm.plasma)

ax2 = fig.add_subplot(2, 2, 2, projection="3d")
plot_surface(ax2, u_true, "Правильное AC: u_t = 0.0001·u_xx − 5·u³ + 5·u",
             cmap=cm.viridis)

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
plot_surface(ax3, u_eq1, "Найденное #1: u_t = 1.4e-7·u_xxx·u_x",
             cmap=cm.cividis)

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
plot_surface(ax4, u_eq2, "Найденное #2: u²·u_t = −0.067·u_tt + 0.332·u_t",
             cmap=cm.magma)

plt.tight_layout()
plt.savefig("ac_surfaces.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------
# 6. Двумерные срезы (более наглядно)
# ----------------------------------------------------------------------
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (Z, name) in zip(axes.ravel(),
                         [(u_data, "Исходные данные"),
                          (u_true, "Правильное AC"),
                          (u_eq1, "Найденное #1"),
                          (u_eq2, "Найденное #2")]):
    im = ax.imshow(Z.T, origin="lower", aspect="auto",
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   cmap="RdBu_r")
    ax.set_title(name)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("ac_surfaces1.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------
# 7. Сравнение ошибок найденных уравнений относительно данных
# ----------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("RMSE (данные vs правильное AC):     ", rmse(u_data, u_true))
print("RMSE (данные vs найденное #1):      ", rmse(u_data, u_eq1))
print("RMSE (данные vs найденное #2):      ", rmse(u_data, u_eq2))