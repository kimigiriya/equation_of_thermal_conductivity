import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def load_params(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                params = json.load(f)
                print(f"Параметры загружены из {filename}")
                return params
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")

    print("Файл параметров не найден. Введите параметры задачи:")
    params = {
        "a_coef": float(input("Введите коэффициент a^2: ")),
        "l": float(input("Введите длину интервала l: ")),
        "T": float(input("Введите максимальное время T: ")),
        "phi_x": input("Начальное условие phi(x) = "),
        "alpha_t": input("Левое граничное условие alpha(t) = "),
        "beta_t": input("Правое граничное условие beta(t) = "),
        "g_x_t": input("Правая часть уравнения g(x,t) = ")
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
        print(f"Параметры сохранены в {filename}")

    return params

def build_funcs(params):
    allowed = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "abs": np.abs,
    }

    def make_func(expr, vars_names):
        code = compile(expr, "<expr>", "eval")
        return lambda **kwargs: eval(code, {"__builtins__": {}}, {**allowed, **kwargs})

    phi_expr = params["phi_x"]
    alpha_expr = params["alpha_t"]
    beta_expr = params["beta_t"]
    g_expr = params["g_x_t"]

    phi = make_func(phi_expr, ["x"])
    alpha = make_func(alpha_expr, ["t"])
    beta = make_func(beta_expr, ["t"])
    g = make_func(g_expr, ["x", "t"])

    return phi, alpha, beta, g

def algorithm(A, B, C, F):
    n = len(A)
    alpha = np.zeros(n)
    betta = np.zeros(n)

    alpha[0] = C[0] / B[0]
    betta[0] = F[0] / B[0]

    for i in range(1, n):
        alpha[i] = -C[i] / (A[i] * alpha[i - 1] - B[i])
        betta[i] = -(F[i] - A[i] * betta[i - 1]) / (A[i] * alpha[i - 1] - B[i])

    y = np.zeros(n)
    y[-1] = betta[-1]
    for i in range(n - 2, -1, -1):
        y[i] = betta[i] - alpha[i] * y[i + 1]

    return y

def solve_explicit_template(params, n, sigma=0.45):
    a2 = float(params["a_coef"])
    l = float(params["l"])
    T = float(params["T"])

    phi, alpha, beta, g = build_funcs(params)

    h = l / n
    tau = sigma * h**2 / a2
    m = int(np.ceil(T / tau))
    tau = T / m
    gamma = a2 * tau / (h**2)

    if gamma > 0.5:
        print(f"gamma = {gamma} > 0.5, схема может быть неустойчивой!")
    else:
        print(f"Так как gamma = {gamma} <= 0.5, то устойчивость выполняется!")

    x = np.linspace(0, l, n + 1)
    t = np.linspace(0, T, m + 1)
    U = np.zeros((m + 1, n + 1), dtype=float)

    for i in range(n + 1):
        U[0, i] = float(phi(x=x[i]))

    for k in range(m + 1):
        U[k, 0] = float(alpha(t=t[k]))
        U[k, -1] = float(beta(t=t[k]))

    for k in range(m):
        for i in range(1, n):
            U[k + 1, i] = (gamma * U[k, i - 1] + (1 - 2 * gamma) * U[k, i] + gamma * U[k, i + 1] + tau * float(g(x=x[i], t=t[k])))
        U[k + 1, 0] = float(alpha(t=t[k + 1]))
        U[k + 1, -1] = float(beta(t=t[k + 1]))

    return x, t, U, h, tau, gamma


def solve_implicit_templat(params, n, sigma):
    a2 = float(params["a_coef"])
    l = float(params["l"])
    T = float(params["T"])

    phi, alpha, beta, g = build_funcs(params)

    h = l / n
    tau = sigma * h**2 / a2
    m = int(np.ceil(T / tau))
    tau = T / m
    gamma = a2 * tau / (h**2)

    x = np.linspace(0, l, n + 1)
    t = np.linspace(0, T, m + 1)

    U = np.zeros((m + 1, n + 1), dtype=float)

    for i in range(n + 1):
        U[0, i] = float(phi(x=x[i]))

    for k in range(m + 1):
        U[k, 0] = float(alpha(t=t[k]))
        U[k, -1] = float(beta(t=t[k]))

    for k in range(1, m + 1):
        A = np.full(n-1, gamma, dtype=float)
        B = np.full(n-1, -(1 + 2*gamma), dtype=float)
        C = np.full(n-1, gamma, dtype=float)
        F = np.zeros(n-1, dtype=float)

        for i in range(1, n):
            F[i - 1] = -U[k - 1, i] - tau * float(g(x=x[i], t=t[k]))

        F[0] -= gamma * U[k, 0]
        F[-1] -= gamma * U[k, -1]

        A[0] = 0
        C[-1] = 0

        U[k, 1:n] = algorithm(A, B, C, F)
        U[k, 0] = float(alpha(t=t[k]))
        U[k, -1] = float(beta(t=t[k]))

    return x, t, U, h, tau, gamma



def plot_3d_surface(x, t, U, title="Решение U(x,t)"):
    X, Tm = np.meshgrid(x, t)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Tm, U, cmap=cm.viridis, linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("U(x,t)")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6, aspect=14)
    plt.tight_layout()
    plt.show()


config_file = "test1.json"
params = load_params(config_file)
n = 50
sigma = 0.45

x, t, U, h, tau, gamma = solve_explicit_template(params, n, sigma=sigma)

print(f"h = {h}")
print(f"tau = {tau}")
print(f"gamma = {gamma}")
print(f"Сетка: {n+1} узлов по x, {len(t)} узлов по t")
plot_3d_surface(x, t, U, title=f"Явная схема, n={n+1}, gamma={gamma}")


x, t, U, h, tau, gamma = solve_implicit_templat(params, n, sigma=sigma)

print(f"h = {h}")
print(f"tau = {tau}")
print(f"gamma = {gamma}")
print(f"Сетка: {n+1} узлов по x, {len(t)} узлов по t")
plot_3d_surface(x, t, U, title=f"Неявная схема, n={n+1}, gamma={gamma}")