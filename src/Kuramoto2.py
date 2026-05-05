import os
import sys
import math
import time

# IMPORTANTE: estas env vars deben fijarse ANTES de importar numpy / numba.
# Cada worker de joblib corre en su propio proceso y dentro de cada proceso
# numpy/BLAS pueden a su vez intentar usar threads (OpenBLAS, MKL, OpenMP).
# Si no los limitamos, con 16 workers de joblib y BLAS multithreaded podriamos
# tener decenas de threads peleandose por los cores: el rendimiento CAE en
# lugar de subir (oversubscription). Al fijarlos a 1, joblib es el unico
# responsable del paralelismo y los threads efectivos son exactamente n_jobs.
# Esto es critico para correr varias instancias del script a la vez sin que
# se machaquen entre si.
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import t as t_dist
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

# Resumen del script: Tarea 2 — red modular (2-3 modulos densamente conectados,
# debilmente interconectados). Estructura paralela a Kuramoto1.
#
# Diferencias relevantes respecto a Tarea 1:
#   - Ecuacion: theta_dot_i = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)
#     (sin 1/N: K es ahora el acoplamiento por enlace).
#
#   - Matriz de adyacencia A: una por run. Las ICs y A se generan una sola vez
#     y se reutilizan para todos los K (Common Random Numbers extendido a la red).
#
#   - r_m(t) por modulo, ademas del R(t) global: calculados en el mismo pase.
#
#   - Eje K logaritmico, opcionalmente concentrado con una t-Student en log(K).
#
#   - Kc experimental: ajuste parabolico al maximo de sigma_R en log(K).
#
#   - Integrador seleccionable: Euler explicito o RK4. Se puede ejecutar
#     un solo metodo o los dos para comparar. RK4 cuesta 4x mas por paso que
#     Euler pero su precision es orden 4 (vs orden 1), lo que en la practica
#     extiende sensiblemente el rango de K usable con el mismo dt.
#
#   - Dos modos de ejecucion seleccionables (booleanos en parametros_simulacion):
#       * FRUSTRACION_SYNC: barrido completo K vs <R>, <r_m>, sigma_R, sigma_rm.
#         Estima Kc local y Kc global. Muchos runs, mucho K.
#       * FRUSTRACION_TEMP: ICs fijas, lista pequena de K, evolucion temporal
#         de R(t) y r_m(t) en una grafica apilada. Permite ver el desfase
#         entre modulos en la zona de metaestabilidad: r_m saturado a 1
#         mientras R(t) oscila, sello de la frustracion modular.
#     Los dos modos pueden activarse a la vez; los resultados van a subcarpetas
#     distintas dentro del mismo run_dir.


def setup_plot_style():
    plt.rcParams.update({
        'font.family'       : 'serif',
        'font.serif'        : ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size'         : 11,
        'axes.labelsize'    : 13,
        'axes.titlesize'    : 14,
        'legend.fontsize'   : 9,
        'xtick.labelsize'   : 10,
        'ytick.labelsize'   : 10,
        'axes.linewidth'    : 1.0,
        'axes.grid'         : True,
        'grid.alpha'        : 0.25,
        'grid.linestyle'    : '--',
        'grid.linewidth'    : 0.5,
        'xtick.direction'   : 'in',
        'ytick.direction'   : 'in',
        'xtick.top'         : True,
        'ytick.right'       : True,
        'xtick.major.size'  : 5,
        'ytick.major.size'  : 5,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon'    : True,
        'legend.framealpha' : 0.95,
        'legend.edgecolor'  : 'black',
        'legend.fancybox'   : False,
        'figure.dpi'        : 100,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
    })


# ----------------------------------------------------------------------------
# Carpetas, ICs y matrices A
# ----------------------------------------------------------------------------

RESULTADOS_BASE = 'resultados\\Tarea2'


def crear_carpeta_resultados(N, num_modules, num_K, num_runs, t_max, desfase_inducido=None):
    os.makedirs(RESULTADOS_BASE, exist_ok=True)
    base = f"N{N}_M{num_modules}_K{num_K}_Runs{num_runs}_t{t_max}"
    if desfase_inducido is not None:
        base += f"_desf{desfase_inducido:.2f}"
    path = os.path.join(RESULTADOS_BASE, base)
    n = 1
    while os.path.exists(path):
        path = os.path.join(RESULTADOS_BASE, f"{base}({n})")
        n += 1
    os.makedirs(path)
    return os.path.abspath(path)

def generar_ICs(num_runs, N, sigma, seed=None):
    rng = np.random.default_rng(seed)
    omegas_IC = rng.normal(0.0, sigma, size=(num_runs, N))
    thetas_IC = rng.uniform(-np.pi, np.pi, size=(num_runs, N))
    return omegas_IC, thetas_IC

def crear_matriz_adyacencia(N, num_modules, n_aristas, p_intra, rng):
    """Bloques de N//num_modules nodos con conectividad intra densa (p_intra)
    y unas pocas aristas inter-modulares (n_aristas por cada par de modulos).
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((N, N), dtype=np.float64)
    module_id = np.zeros(N, dtype=np.int64)

    for i in range(num_modules-1):
        module_id[i*(N//num_modules):(i+1)*(N//num_modules)] = i
    # FIX: el ultimo modulo absorbe el resto (incluido el caso N no divisible)
    module_id[(num_modules-1)*(N//num_modules):] = num_modules-1

    for i in range(N):
        for j in range(i+1, N):
            if module_id[i] == module_id[j]:
                if rng.random() < p_intra:
                    A[i, j] = 1
                    A[j, i] = 1

    for i in range(num_modules):
        for j in range(i+1, num_modules):
            for _ in range(n_aristas):
                a = rng.integers(low=i*(N//num_modules), high=(i+1)*(N//num_modules))
                b = rng.integers(low=j*(N//num_modules), high=(j+1)*(N//num_modules))
                while A[a, b] == 1:
                    a = rng.integers(low=i*(N//num_modules), high=(i+1)*(N//num_modules))
                    b = rng.integers(low=j*(N//num_modules), high=(j+1)*(N//num_modules))
                A[a, b] = A[b, a] = 1

    return A, module_id

def generar_As(num_runs, N, num_modules, n_aristas, p_intra, seed=None):
    rng = np.random.default_rng(seed)
    As = np.zeros((num_runs, N, N), dtype=np.float64)
    module_ids = np.zeros((num_runs, N), dtype=np.int64)
    for r in range(num_runs):
        A, mid = crear_matriz_adyacencia(N, num_modules, n_aristas, p_intra, rng)
        As[r] = A
        module_ids[r] = mid
    return As, module_ids

def guardar_ICs_y_As(omegas_IC, thetas_IC, As, module_ids, run_dir):
    ruta = os.path.join(run_dir, 'condiciones_iniciales.npz')
    np.savez_compressed(ruta, omegas=omegas_IC, thetas=thetas_IC, As=As, module_ids=module_ids)
    return ruta

def _ruta(directorio, nombre):
    os.makedirs(directorio, exist_ok=True)
    return os.path.join(directorio, nombre)

def guardar_params_txt(run_dir, params_dict):
    with open(os.path.join(run_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write("Parametros de la ejecucion\n")
        f.write("=" * 40 + "\n")
        for k, v in params_dict.items():
            f.write(f"{k:25s} = {v}\n")

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)

def iniciar_log(run_dir):
    log_path = os.path.join(run_dir, 'log.txt')
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
    log_file.write(f"Log de ejecucion - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.flush()
    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout_orig, log_file)
    sys.stderr = Tee(stderr_orig, log_file)
    return log_file, stdout_orig, stderr_orig

def cerrar_log(log_file, stdout_orig, stderr_orig):
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    log_file.close()

# ----------------------------------------------------------------------------
# Sistema de Kuramoto en red modular
# ----------------------------------------------------------------------------

class Gaussiana: pass
class Lorentziana: pass

class KuramotoSystem:

    def __init__(self, N: int, num_modules: int, steps: int, dt: float):
        self.N           = N
        self.num_modules = num_modules
        self.steps       = steps
        self.dt          = dt

        self.R   = np.zeros(steps + 1, dtype=np.float64)
        self.psi = np.zeros(steps + 1, dtype=np.float64)
        self.r_m = np.zeros((num_modules, steps + 1), dtype=np.float64)

        self.omega      = np.zeros(N, dtype=np.float64)
        self.theta_curr = np.zeros(N, dtype=np.float64)
        self.theta_next = np.zeros(N, dtype=np.float64)

        self.A         = np.zeros((N, N), dtype=np.float64)
        self.module_id = np.zeros(N, dtype=np.int64)

    

    def initialize(self, distr, sigma=1.0, omega=None, theta_0=None, A=None, module_id=None):
        if omega is None:
            if distr is Lorentziana:
                self.omega = sigma * np.random.standard_cauchy(self.N) # Distribución lorentziana
            elif distr is Gaussiana:
                self.omega = np.random.normal(0, sigma, self.N) # Distribución Gaussiana
        else:
            self.omega = np.asarray(omega, dtype=np.float64).copy()
        if theta_0 is None:
            self.theta_curr = np.random.uniform(-np.pi, np.pi, self.N)
        else:
            self.theta_curr = np.asarray(theta_0, dtype=np.float64).copy()
        if A is None:
            raise ValueError("Hay que pasar la matriz de adyacencia A.")
        if module_id is None:
            raise ValueError("Hay que pasar module_id.")
        self.A         = np.ascontiguousarray(A, dtype=np.float64)
        self.module_id = np.asarray(module_id, dtype=np.int64).copy()

    def run(self, K: float, method: str = 'rk4'):
        """method: 'euler' o 'rk4'."""
        if method == 'euler':
            _integrar_euler(self.theta_curr, self.theta_next, self.omega, self.A, self.module_id, K, self.dt, self.steps, self.R, self.psi, self.r_m)
        elif method == 'rk4':
            _integrar_rk4(self.theta_curr, self.theta_next, self.omega, self.A, self.module_id, K, self.dt, self.steps, self.R, self.psi, self.r_m)
        else:
            raise ValueError(f"Metodo desconocido: '{method}'. Usa 'euler' o 'rk4'.")

    @property
    def R_mean(self) -> float:
        n_trans = self.steps // 4
        return float(np.mean(self.R[n_trans:]))

    @property
    def R_std(self) -> float:
        n_trans = self.steps // 4
        return float(np.std(self.R[n_trans:]))

    @property
    def r_m_mean(self) -> np.ndarray:
        n_trans = self.steps // 4
        return np.mean(self.r_m[:, n_trans:], axis=1)

    @property
    def r_m_std(self) -> np.ndarray:
        n_trans = self.steps // 4
        return np.std(self.r_m[:, n_trans:], axis=1)

# ----------------------------------------------------------------------------
# Helpers numericos: rhs y observables
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _eval_rhs(theta, omega, A, K, rhs):
    """Evalua el rhs de Kuramoto in-place sobre 'rhs'.

        rhs[i] = omega[i] + K * sum_j A[i,j] * sin(theta[j] - theta[i])

    Esto es lo que cuesta O(N^2) y se invoca 1 vez por paso (Euler) o
    4 veces por paso (RK4). Es el bottleneck; numba lo deberia inlinar.
    """
    N = theta.shape[0]
    for i in range(N):
        ti = theta[i]
        c = 0.0
        for j in range(N):
            c += A[i, j] * math.sin(theta[j] - ti)
        rhs[i] = omega[i] + K * c

@njit(fastmath=True, cache=True)
def _calcular_observables(theta, module_id, module_size, R, psi, r_m, t, re_m, im_m):
    """Calcula R(t), psi(t) y r_m(t) en un solo pase sobre theta."""
    N = theta.shape[0]
    M = r_m.shape[0]
    re, im = 0.0, 0.0
    for m in range(M):
        re_m[m] = 0.0
        im_m[m] = 0.0
    for j in range(N):
        cj = math.cos(theta[j])
        sj = math.sin(theta[j])
        re += cj
        im += sj
        mid = module_id[j]
        re_m[mid] += cj
        im_m[mid] += sj
    R[t]   = math.sqrt(re*re + im*im) / N
    psi[t] = math.atan2(im, re)
    for m in range(M):
        r_m[m, t] = math.sqrt(re_m[m]*re_m[m] + im_m[m]*im_m[m]) / module_size[m]

# ----------------------------------------------------------------------------
# Integrador 1: Euler explicito
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _integrar_euler(theta_curr, theta_next, omega, A, module_id, K, dt, steps, R, psi, r_m):
    """Euler explicito.

    Coste: 1 evaluacion de rhs por paso.
    Cota de estabilidad lineal: dt * K * lambda_max(L) < 2.
    Precision: O(dt).
    """
    N = theta_curr.shape[0]
    M = r_m.shape[0]

    module_size = np.zeros(M, dtype=np.float64)
    for i in range(N):
        module_size[module_id[i]] += 1.0

    re_m = np.zeros(M, dtype=np.float64)
    im_m = np.zeros(M, dtype=np.float64)
    rhs  = np.zeros(N, dtype=np.float64)

    for t in range(steps):
        _calcular_observables(theta_curr, module_id, module_size, R, psi, r_m, t, re_m, im_m)

        _eval_rhs(theta_curr, omega, A, K, rhs)
        for i in range(N):
            theta_next[i] = theta_curr[i] + dt * rhs[i]

        for i in range(N):
            theta_curr[i] = theta_next[i]

    _calcular_observables(theta_curr, module_id, module_size, R, psi, r_m, steps, re_m, im_m)

# ----------------------------------------------------------------------------
# Integrador 2: Runge-Kutta 4
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _integrar_rk4(theta_curr, theta_next, omega, A, module_id, K, dt, steps, R, psi, r_m):
    """Runge-Kutta clasico de 4 etapas.

        k1 = f(theta_n)
        k2 = f(theta_n + dt/2 * k1)
        k3 = f(theta_n + dt/2 * k2)
        k4 = f(theta_n + dt   * k3)
        theta_{n+1} = theta_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Coste: 4 evaluaciones de rhs por paso (4x mas que Euler).
    Cota de estabilidad lineal: dt * K * lambda_max(L) < 2.78.
    Precision: O(dt^4).

    El factor 2.78/2 ~ 1.4 mejora poco la cota lineal, pero la ganancia
    practica viene de la precision orden-4: la integracion permanece
    controlada en regimenes en los que Euler diverge incluso por encima
    del bound lineal estricto, gracias a la saturacion no lineal del seno.
    """
    N = theta_curr.shape[0]
    M = r_m.shape[0]

    module_size = np.zeros(M, dtype=np.float64)
    for i in range(N):
        module_size[module_id[i]] += 1.0

    re_m = np.zeros(M, dtype=np.float64)
    im_m = np.zeros(M, dtype=np.float64)

    # Buffers RK4 (reusados en cada paso)
    k1        = np.zeros(N, dtype=np.float64)
    k2        = np.zeros(N, dtype=np.float64)
    k3        = np.zeros(N, dtype=np.float64)
    k4        = np.zeros(N, dtype=np.float64)
    theta_tmp = np.zeros(N, dtype=np.float64)

    dt_2 = 0.5 * dt
    dt_6 = dt / 6.0

    for t in range(steps):
        _calcular_observables(theta_curr, module_id, module_size, R, psi, r_m, t, re_m, im_m)

        _eval_rhs(theta_curr, omega, A, K, k1)

        for i in range(N):
            theta_tmp[i] = theta_curr[i] + dt_2 * k1[i]
        _eval_rhs(theta_tmp, omega, A, K, k2)

        for i in range(N):
            theta_tmp[i] = theta_curr[i] + dt_2 * k2[i]
        _eval_rhs(theta_tmp, omega, A, K, k3)

        for i in range(N):
            theta_tmp[i] = theta_curr[i] + dt * k3[i]
        _eval_rhs(theta_tmp, omega, A, K, k4)

        for i in range(N):
            theta_next[i] = theta_curr[i] + dt_6 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])

        for i in range(N):
            theta_curr[i] = theta_next[i]

    _calcular_observables(theta_curr, module_id, module_size, R, psi, r_m, steps, re_m, im_m)

# ----------------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------------

def Simulacion_Kuramoto(N, num_modules, K, sigma, dt, t_max, A, module_id, omega=None, theta_0=None, method='rk4'):
    num_pasos = int(t_max / dt)
    sys = KuramotoSystem(N=N, num_modules=num_modules, steps=num_pasos, dt=dt)
    sys.initialize(sigma=sigma, omega=omega, theta_0=theta_0, A=A, module_id=module_id)
    sys.run(K=K, method=method)
    return sys

def _una_simulacion_indexada(j, r, N, num_modules, K, sigma, dt, t_max, omega_ic, theta0_ic, A, module_id, method):
    sys = Simulacion_Kuramoto(N, num_modules, K, sigma, dt, t_max, A=A, module_id=module_id, omega=omega_ic, theta_0=theta0_ic, method=method)
    return (j, r, sys.R_mean, sys.R_std, sys.r_m_mean, sys.r_m_std)

def barrido_completo(N, num_modules, K_values, T_per_K, num_runs, sigma, dt, omegas_IC, thetas_IC, As, module_ids, method='rk4', n_jobs=-1):
    num_K = len(K_values)

    R_means       = np.zeros(num_K)
    R_stds        = np.zeros(num_K)
    R_mean_stds   = np.zeros(num_K)
    rm_means      = np.zeros((num_K, num_modules))
    rm_stds       = np.zeros((num_K, num_modules))
    rm_mean_stds  = np.zeros((num_K, num_modules))

    total_ut = T_per_K.sum() * num_runs
    print(f"Barrido [{method.upper()}]: {num_K} K x {num_runs} runs = "
          f"{num_K * num_runs} simulaciones")
    print(f"  Total simulado: {total_ut:.0f} u.t. "
          f"({total_ut*int(1/dt):.2e} pasos)")
    print(f"  Workers (n_jobs): {n_jobs}\n")

    tareas = []
    for j, K in enumerate(K_values):
        t_max_j = float(T_per_K[j])
        for r in range(num_runs):
            tareas.append((j, r, N, num_modules, K, sigma, dt, t_max_j, omegas_IC[r], thetas_IC[r], As[r], module_ids[r], method))
    tareas.sort(key=lambda tarea: -tarea[7])

    print(f"Lanzando {len(tareas)} simulaciones en paralelo...")
    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_una_simulacion_indexada)(*tarea) for tarea in tareas
    )

    means_por_K    = {j: [] for j in range(num_K)}
    stds_por_K     = {j: [] for j in range(num_K)}
    rm_means_por_K = {j: [] for j in range(num_K)}
    rm_stds_por_K  = {j: [] for j in range(num_K)}

    for (j, r, r_mean, r_std, rm_mean, rm_std) in resultados:
        means_por_K[j].append(r_mean)
        stds_por_K[j].append(r_std)
        rm_means_por_K[j].append(rm_mean)
        rm_stds_por_K[j].append(rm_std)

    for j in range(num_K):
        R_means[j]     = np.mean(means_por_K[j])
        R_stds[j]      = np.mean(stds_por_K[j])
        R_mean_stds[j] = np.std(means_por_K[j])
        rm_means_arr = np.asarray(rm_means_por_K[j])
        rm_stds_arr  = np.asarray(rm_stds_por_K[j])
        rm_means[j]      = np.mean(rm_means_arr, axis=0)
        rm_stds[j]       = np.mean(rm_stds_arr,  axis=0)
        rm_mean_stds[j]  = np.std(rm_means_arr,  axis=0)

    return R_means, R_stds, R_mean_stds, rm_means, rm_stds, rm_mean_stds

# ----------------------------------------------------------------------------
# Kc experimental y K-grid log
# ----------------------------------------------------------------------------

def Kc_experimental(K_values, R_stds, window=3, log=True):
    K_values = np.asarray(K_values)
    R_stds   = np.asarray(R_stds)
    idx      = int(np.argmax(R_stds))

    lo = max(0, idx - window)
    hi = min(len(K_values), idx + window + 1)
    if hi - lo < 3:
        return float(K_values[idx])

    K_win = K_values[lo:hi]
    R_win = R_stds[lo:hi]
    x = np.log(K_win) if log else K_win
    a, b, _ = np.polyfit(x, R_win, 2)
    if a >= 0:
        return float(K_values[idx])
    x_vertex = -b / (2 * a)
    if x_vertex < x[0] or x_vertex > x[-1]:
        return float(K_values[idx])
    return float(np.exp(x_vertex)) if log else float(x_vertex)

# ----------------------------------------------------------------------------
# Ajuste de R(K) a la curva tipo Kuramoto generalizada
# ----------------------------------------------------------------------------

def R_modelo(K, Kc, alpha, R_inf):
    """Modelo Kuramoto generalizado para UNA transicion.

        R(K) = R_inf * sqrt( max(0, 1 - (Kc/K)^alpha) )

    alpha=1 recupera la forma Lorentziana exacta. Para gaussiana sale
    alpha en ~[1.5, 2.5]. R_inf permite saturacion < 1 por finite-size.
    """
    K = np.asarray(K, dtype=np.float64)
    arg = 1.0 - (Kc / K)**alpha
    return R_inf * np.sqrt(np.clip(arg, 0.0, None))

def ajustar_R_simple(K_data, R_mean, R_err, Kc_guess,
                     K_min_fit=None, K_max_fit=None, noise_floor=None,
                     R_inf_max=1.05):
    """Ajusta R_modelo a un conjunto (K, R_mean +/- R_err) por minimos cuadrados
    pesados. Devuelve (popt, perr) con popt = [Kc, alpha, R_inf].

    Filtros opcionales antes del ajuste:
      - K_min_fit / K_max_fit: recorta el rango de K (util para aislar una
        transicion concreta cuando hay varias).
      - noise_floor: descarta puntos con R_mean < noise_floor (evita sesgar
        Kc con la cola incoherente, donde R ~ 1/sqrt(N) por finite-size).
    """
    K_data = np.asarray(K_data, dtype=np.float64)
    R_mean = np.asarray(R_mean, dtype=np.float64)
    R_err  = np.asarray(R_err,  dtype=np.float64)

    mask = np.ones_like(R_mean, dtype=bool)
    if K_min_fit is not None:
        mask &= (K_data >= K_min_fit)
    if K_max_fit is not None:
        mask &= (K_data <= K_max_fit)
    if noise_floor is not None:
        mask &= (R_mean > noise_floor)

    if mask.sum() < 4:
        return (np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]))

    K_fit = K_data[mask]
    R_fit = R_mean[mask]
    E_fit = np.maximum(R_err[mask], 1e-6)   # curve_fit exige sigma > 0

    p0     = [Kc_guess, 2.0, 1.0]
    bounds = ([max(K_fit.min() * 0.1, 1e-6), 0.5, 0.3],
              [K_fit.max() * 3.0,             6.0, R_inf_max])

    try:
        popt, pcov = curve_fit(R_modelo, K_fit, R_fit,
                               p0=p0, bounds=bounds,
                               sigma=E_fit, absolute_sigma=True,
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"    AVISO: ajuste fallo ({e})")
        popt = np.array([np.nan]*3)
        perr = np.array([np.nan]*3)

    return popt, perr

def ajustar_todas_las_curvas(K_values, R_means, R_mean_stds, R_stds,
                              rm_means, rm_mean_stds, rm_stds,
                              num_modules, N):
    """Orquesta los ajustes para todas las curvas.

    Devuelve dict con keys 'global' y 'm0', 'm1', ..., cada uno con
    {'popt': [Kc, alpha, R_inf], 'perr': [...]}.
    """
    # Piso de ruido finite-size: ~1.5/sqrt(N_efectivo)
    noise_global = 1.5 / math.sqrt(N)
    noise_modulo = 1.5 / math.sqrt(N // num_modules)

    fits = {}

    # --- Por modulo: una transicion limpia ---
    Kc_locals = []
    for m in range(num_modules):
        Kc0 = Kc_experimental(K_values, rm_stds[:, m], log=True)
        popt, perr = ajustar_R_simple(
            K_values, rm_means[:, m], rm_mean_stds[:, m],
            Kc_guess=Kc0, noise_floor=noise_modulo,
        )
        fits[f'm{m}'] = {'popt': popt, 'perr': perr}
        if np.isfinite(popt[0]):
            Kc_locals.append(popt[0])

    # --- Global: dos transiciones, aislamos la global ---
    Kc0_glob = Kc_experimental(K_values, R_stds, log=True)
    Kc_loc_med = np.median(Kc_locals) if Kc_locals else None

    # Si hay separacion de escalas, ajustamos solo K > 1.5*Kc_local
    K_min_fit_glob = None
    if Kc_loc_med is not None and Kc0_glob > 2.5 * Kc_loc_med:
        K_min_fit_glob = 1.5 * Kc_loc_med

    popt, perr = ajustar_R_simple(
        K_values, R_means, R_mean_stds,
        Kc_guess=Kc0_glob,
        K_min_fit=K_min_fit_glob,
        noise_floor=(noise_global if K_min_fit_glob is None else None),
    )
    fits['global'] = {'popt': popt, 'perr': perr,
                      'K_min_fit': K_min_fit_glob}

    # --- Resumen por consola ---
    print("\n  Ajustes R_modelo (Kc, alpha, R_inf) +/- 1-sigma:")
    print(f"  {'curva':>8} | {'Kc':>12} | {'alpha':>10} | {'R_inf':>10} | nota")
    for m in range(num_modules):
        p, e = fits[f'm{m}']['popt'], fits[f'm{m}']['perr']
        print(f"  {'r_'+str(m+1):>8} | {p[0]:>6.4g} +/-{e[0]:>5.2g} "
              f"| {p[1]:>5.3f} +/-{e[1]:>4.2g} "
              f"| {p[2]:>5.3f} +/-{e[2]:>4.2g} |")
    p, e = fits['global']['popt'], fits['global']['perr']
    nota = (f"K > {K_min_fit_glob:.3g}" if K_min_fit_glob else "rango completo")
    print(f"  {'R glob':>8} | {p[0]:>6.4g} +/-{e[0]:>5.2g} "
          f"| {p[1]:>5.3f} +/-{e[1]:>4.2g} "
          f"| {p[2]:>5.3f} +/-{e[2]:>4.2g} | {nota}")

    return fits

def K_values_log_tstudent(num_K, K_min, K_max, K_center=None, width_factor=0.5, df=2):
    log_K_min = np.log(K_min)
    log_K_max = np.log(K_max)
    if K_center is None:
        return np.exp(np.linspace(log_K_min, log_K_max, num_K))
    log_K_center = np.log(K_center)
    width = width_factor * max(abs(log_K_center), 1.0)
    q_min = t_dist.cdf(log_K_min, df=df, loc=log_K_center, scale=width)
    q_max = t_dist.cdf(log_K_max, df=df, loc=log_K_center, scale=width)
    qs    = np.linspace(q_min, q_max, num_K)
    log_K = t_dist.ppf(qs, df=df, loc=log_K_center, scale=width)
    return np.exp(log_K)

def t_max_per_K_log(K_values, K_center, t_max_base, t_max_peak, width_factor=0.5, df=2):
    K_values = np.asarray(K_values, dtype=np.float64)
    if K_center is None:
        return np.full_like(K_values, t_max_base)
    log_K        = np.log(K_values)
    log_K_center = np.log(K_center)
    width        = width_factor * max(abs(log_K_center), 1.0)
    pdf     = t_dist.pdf(log_K,        df=df, loc=log_K_center, scale=width)
    pdf_max = t_dist.pdf(log_K_center, df=df, loc=log_K_center, scale=width)
    weight  = pdf / pdf_max
    return t_max_base + (t_max_peak - t_max_base) * weight

# ----------------------------------------------------------------------------
# Visualizacion de la matriz de adyacencia
# ----------------------------------------------------------------------------

def stats_matriz_adyacencia(A, module_id):
    num_modules = int(module_id.max()) + 1
    densidades = np.zeros((num_modules, num_modules))
    for i in range(num_modules):
        idx_i = np.where(module_id == i)[0]
        for j in range(num_modules):
            idx_j = np.where(module_id == j)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            sub = A[np.ix_(idx_i, idx_j)]
            if i == j:
                n_pares   = len(idx_i) * (len(idx_i) - 1) / 2
                n_aristas = sub.sum() / 2
            else:
                n_pares   = len(idx_i) * len(idx_j)
                n_aristas = sub.sum()
            densidades[i, j] = n_aristas / n_pares if n_pares > 0 else 0.0
    return densidades

def plot_matriz_adyacencia(A, module_id, save_path, title=None):
    setup_plot_style()
    N           = A.shape[0]
    num_modules = int(module_id.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    ax.imshow(A, cmap='Greys', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    sizes = np.bincount(module_id, minlength=num_modules)
    cum   = np.cumsum(sizes)
    for c in cum[:-1]:
        ax.axhline(c - 0.5, color='red', linewidth=1.0, alpha=0.8)
        ax.axvline(c - 0.5, color='red', linewidth=1.0, alpha=0.8)
    centers, start = [], 0
    for m in range(num_modules):
        centers.append(start + sizes[m] / 2 - 0.5 if sizes[m] > 0 else np.nan)
        start += sizes[m]
    visible = [(c, m) for c, m in zip(centers, range(num_modules)) if not np.isnan(c)]
    if visible:
        ticks, labs = zip(*[(c, f'M{m}') for c, m in visible])
        ax.set_xticks(ticks); ax.set_xticklabels(labs)
        ax.set_yticks(ticks); ax.set_yticklabels(labs)

    n_edges = int(A.sum() / 2)
    if title is None:
        title = fr'Matriz de adyacencia  $N={N}$, $|E|={n_edges}$'
    ax.set_title(title)

    ax = axes[1]
    densidades = stats_matriz_adyacencia(A, module_id)
    im = ax.imshow(densidades, cmap='viridis', aspect='equal', vmin=0, vmax=1)
    for i in range(num_modules):
        for j in range(num_modules):
            ax.text(j, i, f'{densidades[i,j]:.3g}', ha='center', va='center', color='white' if densidades[i,j] < 0.5 else 'black', fontsize=11)
    ax.set_xticks(range(num_modules))
    ax.set_xticklabels([f'M{m}' for m in range(num_modules)])
    ax.set_yticks(range(num_modules))
    ax.set_yticklabels([f'M{m}' for m in range(num_modules)])
    ax.set_title('Densidad de aristas por bloque')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    txt = '  '.join([fr'$|M_{m}|={sizes[m]}$' for m in range(num_modules)])
    fig.text(0.5, 0.01, txt, ha='center', fontsize=10, style='italic', alpha=0.8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path

# ----------------------------------------------------------------------------
# Plots de barrido (modo SYNC)
# ----------------------------------------------------------------------------

def _draw_R_vs_K(ax, K_values, R_means, R_mean_stds, rm_means, rm_mean_stds,
                 num_modules, color_global, colors_modules, fits=None):
    # Curva fina para los ajustes (geometrica porque K es log)
    K_fine = np.geomspace(K_values.min(), K_values.max(), 500)

    for m in range(num_modules):
        ax.fill_between(K_values, rm_means[:, m] - rm_mean_stds[:, m],
                                  rm_means[:, m] + rm_mean_stds[:, m],
                        color=colors_modules[m], alpha=0.15)
        ax.plot(K_values, rm_means[:, m], marker='o', markersize=3,
                linewidth=0, color=colors_modules[m],
                label=fr'$\langle r_{{{m+1}}} \rangle$')
        # Ajuste
        if fits is not None and f'm{m}' in fits:
            popt = fits[f'm{m}']['popt']
            if np.isfinite(popt[0]):
                ax.plot(K_fine, R_modelo(K_fine, *popt),
                        linestyle='-', linewidth=1.2,
                        color=colors_modules[m], alpha=0.8)

    ax.fill_between(K_values, R_means - R_mean_stds, R_means + R_mean_stds,
                    color=color_global, alpha=0.25)
    ax.plot(K_values, R_means, marker='o', markersize=4, linewidth=0,
            color=color_global, label=r'$\langle R \rangle$ (global)')
    if fits is not None and 'global' in fits:
        popt = fits['global']['popt']
        if np.isfinite(popt[0]):
            # Si el fit fue segmentado, solo dibujarlo en su rango
            K_lo = fits['global'].get('K_min_fit') or K_fine.min()
            mask_fine = K_fine >= K_lo
            ax.plot(K_fine[mask_fine], R_modelo(K_fine[mask_fine], *popt),
                    linestyle='-', linewidth=1.6,
                    color=color_global, alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel(r'Acoplamiento $K$  (escala log)')
    ax.set_ylabel(r'Parametros de orden')
    ax.set_title(r'Sincronizacion local vs global (red modular)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')

def _draw_sigmaR_vs_K(ax, K_values, R_stds, rm_stds, num_modules, color_global, colors_modules):
    for m in range(num_modules):
        Kc_m = Kc_experimental(K_values, rm_stds[:, m], log=True)
        ax.plot(K_values, rm_stds[:, m], marker='o', markersize=3, linewidth=1.2, color=colors_modules[m], label=fr'$\sigma_{{r_{{{m+1}}}}}$   $K_c^{{({m+1})}} = {Kc_m:.3g}$')
        ax.axvline(Kc_m, color=colors_modules[m], linestyle=':', linewidth=1.0, alpha=0.6)
    Kc_g = Kc_experimental(K_values, R_stds, log=True)
    ax.plot(K_values, R_stds, marker='o', markersize=4, linewidth=1.8, color=color_global, label=fr'$\sigma_R$ (global)   $K_c = {Kc_g:.3g}$')
    ax.axvline(Kc_g, color=color_global, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'Acoplamiento $K$  (escala log)')
    ax.set_ylabel(r'Desviacion estandar')
    ax.set_title(r'Metaestabilidad: fluctuaciones del parametro de orden')
    ax.legend(loc='upper right')

def _add_info_box(ax, N, num_modules, num_runs, loc='bottom'):
    y, va = (0.05, 'bottom') if loc == 'bottom' else (0.95, 'top')
    ax.text(0.98, y, fr'$N = {N}$, $M = {num_modules}$, runs $= {num_runs}$', transform=ax.transAxes, ha='right', va=va, fontsize=9, style='italic', alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))

def plot_R_vs_K(K_values, R_means, R_mean_stds, rm_means, rm_mean_stds, num_modules, N, num_runs, save_dir, suffix='', fits=None):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    color_global   = 'k'
    colors_modules = plt.cm.viridis(np.linspace(0.2, 0.85, num_modules))
    _draw_R_vs_K(ax, K_values, R_means, R_mean_stds, rm_means, rm_mean_stds, num_modules, color_global, colors_modules, fits=fits)
    _add_info_box(ax, N, num_modules, num_runs, loc='bottom')
    fig.savefig(_ruta(save_dir, f'R_vs_K{suffix}.png'))
    plt.close(fig)

def plot_combined(K_values, R_means, R_stds, R_mean_stds, rm_means, rm_stds, rm_mean_stds, num_modules, N, num_runs, save_dir, suffix='', fits=None):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    color_global   = 'k'
    colors_modules = plt.cm.viridis(np.linspace(0.2, 0.85, num_modules))
    _draw_R_vs_K(ax1, K_values, R_means, R_mean_stds, rm_means, rm_mean_stds, num_modules, color_global, colors_modules, fits=fits)
    _add_info_box(ax1, N, num_modules, num_runs, loc='bottom')
    _draw_sigmaR_vs_K(ax2, K_values, R_stds, rm_stds, num_modules, color_global, colors_modules)
    _add_info_box(ax2, N, num_modules, num_runs, loc='top')
    fig.tight_layout()
    fig.savefig(_ruta(save_dir, f'combinado{suffix}.png'))
    plt.close(fig)

def plot_sigmaR_vs_K(K_values, R_stds, rm_stds, num_modules, N, num_runs, save_dir, suffix=''):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    color_global   = 'k'
    colors_modules = plt.cm.viridis(np.linspace(0.2, 0.85, num_modules))
    _draw_sigmaR_vs_K(ax, K_values, R_stds, rm_stds, num_modules, color_global, colors_modules)
    _add_info_box(ax, N, num_modules, num_runs, loc='top')
    fig.savefig(_ruta(save_dir, f'sigmaR_vs_K{suffix}.png'))
    plt.close(fig)

# ----------------------------------------------------------------------------
# Plot de comparacion entre metodos (Euler vs RK4)
# ----------------------------------------------------------------------------

def plot_comparacion_metodos(K_values, resultados_por_metodo, num_modules, N, num_runs, save_dir):
    """Compara <R>(K) y sigma_R(K) entre Euler y RK4 en una sola figura.

    El contraste entre las dos curvas hace patente donde Euler diverge
    (saltos espurios en sigma_R, caida no fisica de <R>) y donde RK4
    sigue siendo fiable.
    """
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    estilos = {'euler': dict(linestyle='--', marker='s', markersize=3, linewidth=1.4),'rk4'  : dict(linestyle='-',  marker='o', markersize=3, linewidth=1.6)}
    colores = {'euler': '#d62728', 'rk4': '#1f77b4'}

    for metodo, datos in resultados_por_metodo.items():
        R_means, _, R_mean_stds, _, _, _ = datos
        ax1.fill_between(K_values, R_means - R_mean_stds, R_means + R_mean_stds, color=colores[metodo], alpha=0.15)
        ax1.plot(K_values, R_means, color=colores[metodo], label=fr'$\langle R \rangle$ — {metodo.upper()}', **estilos[metodo])
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Acoplamiento $K$  (escala log)')
    ax1.set_ylabel(r'$\langle R \rangle$  (global)')
    ax1.set_title(r'Comparacion: $\langle R \rangle$ vs $K$ por metodo')
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc='lower right')
    _add_info_box(ax1, N, num_modules, num_runs, loc='bottom')

    for metodo, datos in resultados_por_metodo.items():
        _, R_stds, _, _, _, _ = datos
        Kc_m = Kc_experimental(K_values, R_stds, log=True)
        ax2.plot(K_values, R_stds, color=colores[metodo], label=fr'$\sigma_R$ — {metodo.upper()}   $K_c = {Kc_m:.3g}$', **estilos[metodo])
        ax2.axvline(Kc_m, color=colores[metodo], linestyle=':', linewidth=1.0, alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Acoplamiento $K$  (escala log)')
    ax2.set_ylabel(r'$\sigma_R$  (global)')
    ax2.set_title(r'Comparacion: $\sigma_R$ vs $K$ por metodo')
    ax2.legend(loc='upper right')
    _add_info_box(ax2, N, num_modules, num_runs, loc='top')

    fig.tight_layout()
    fig.savefig(_ruta(save_dir, 'comparacion_metodos.png'))
    plt.close(fig)

# ----------------------------------------------------------------------------
# Plot de evolucion temporal (modo TEMP)
# ----------------------------------------------------------------------------

def plot_evolucion_temporal(K_list, series_por_K, num_modules, dt, N, save_dir, transient_frac=0.25, max_points=3000, method_name='RK4', suffix=''):
    """Evolucion temporal de R(t) y r_m(t) para varios K, en una columna apilada.

    Un panel por K. En cada panel:
      - r_m(t) (uno por modulo, colores viridis tenues)
      - R(t)   (negro, mas grueso)
      - Region transitoria sombreada (primer transient_frac * t_max)

    Las series se submuestrean a max_points para que el PNG sea legible.
    """
    setup_plot_style()
    n_K = len(K_list)

    fig, axes = plt.subplots(n_K, 1, figsize=(10, 2.6 * n_K), sharex=True)
    if n_K == 1:
        axes = [axes]

    colors_modules = plt.cm.viridis(np.linspace(0.25, 0.85, num_modules))

    for idx, (K, series) in enumerate(zip(K_list, series_por_K)):
        ax = axes[idx]
        R   = series['R']
        r_m = series['r_m']
        n_pts = len(R)
        t_array = np.arange(n_pts) * dt

        # Submuestreo solo para visualizacion
        stride = max(1, n_pts // max_points)
        t_p   = t_array[::stride]
        R_p   = R[::stride]
        r_m_p = r_m[:, ::stride]

        # Sombrear transitorio
        t_trans = t_array[-1] * transient_frac
        ax.axvspan(0, t_trans, color='gray', alpha=0.08)

        # r_m(t)
        for m in range(num_modules):
            ax.plot(t_p, r_m_p[m], color=colors_modules[m], linewidth=0.5,alpha=0.85, label=fr'$r_{{{m+1}}}(t)$')
        # R(t)
        ax.plot(t_p, R_p, color='k', linewidth=0.75, label=r'$R(t)$')

        ax.set_ylabel('parametros')
        ax.set_ylim(-0.02, 1.02)
        ax.text(0.985, 0.92, fr'$K = {K:.3g}$',transform=ax.transAxes, ha='right', va='top',fontsize=11, fontweight='bold',bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

        # Anotar <R> y sigma_R del estacionario en cada panel
        n_trans = n_pts // 4
        R_mean = float(np.mean(R[n_trans:]))
        R_std  = float(np.std(R[n_trans:]))
        ax.text(0.015, 0.92,fr'$\langle R \rangle = {R_mean:.3f}$, $\sigma_R = {R_std:.3f}$',transform=ax.transAxes, ha='left', va='top',fontsize=9, alpha=0.85,bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.7))

    axes[-1].set_xlabel(r'tiempo $t$ (u.t.)')
    # Legend solo en el primer panel
    axes[0].legend(loc='lower right', ncol=num_modules + 1, fontsize=9)

    fig.suptitle(fr'Evolucion temporal — red modular ({method_name}, ICs fijas), $N={N}$, $M={num_modules}$',fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(_ruta(save_dir, f'evolucion_temporal{suffix}.png'),bbox_inches='tight')
    plt.close(fig)

# ----------------------------------------------------------------------------
# Parametros y setup
# ----------------------------------------------------------------------------

# Soy muy raro, pero me gusta tener las cosas organizadas, así que vamos a crear una función auxiliar de parámetros, que
# nos devuelva absolutamente todos los parámetros necesarios para la simulación.

def parametros_simulacion():
    # --------------------- Parametros del sistema ---------------------
    N           = 128
    num_modules = 2
    sigma       = 1.0
    dt          = 0.001
    n_aristas   = 1          # cuello de botella inter-modular
    p_intra     = 1.0        # bloques fully-connected

    # --------------------- Barrido de K (log scale) — modo SYNC ---------------------
    num_K        = 10
    K_min        = 5e-3
    K_max        = 0.95 * (2 / dt * (N//num_modules + 1))           # Aquí estamos limitando las K, para que no nos salgamos del rango estable del modelo.
    K_center     = None
    width_factor = 0.5

    # --------------------- t_max adaptativo — modo SYNC ---------------------
    t_max_base = 300.0
    t_max_peak = 1500.0

    # --------------------- Runs — modo SYNC ---------------------
    num_runs = 50

    # --------------------- Metodos de integracion ---------------------
    # 'euler' | 'rk4' | ['euler', 'rk4']
    methods = ['euler']
    if isinstance(methods, str):
        methods = [methods]

    # --------------------- Paralelismo ---------------------
    # Numero de workers de joblib. Ojo: cada worker es un proceso aparte y
    # las env vars de OMP/MKL/OpenBLAS estan ya fijadas a 1 (ver cabecera del
    # script), asi que aqui mandan los procesos.
    #   -1 -> todos los cores logicos disponibles
    #    n -> exactamente n cores
    # Override desde shell sin tocar el codigo:
    #   PowerShell: $env:KURAMOTO_N_JOBS=12; python Kuramoto2.py
    #   Bash:       KURAMOTO_N_JOBS=12 python Kuramoto2.py
    n_jobs_default = 16
    n_jobs = int(os.environ.get('KURAMOTO_N_JOBS', n_jobs_default))

    # --------------------- Tipo de simulacion ---------------------
    FRUSTRACION_SYNC = True     # barrido completo en K (caro)
    FRUSTRACION_TEMP = False      # ICs fijas, evolucion temporal en varios K

    # --------------------- Parametros del modo TEMP ---------------------
    # Lista pequena de K representativos. Sugerencia: incluir un K bajo
    # (incoherencia), un K post-Kc_local (sync local pero R oscila), uno o dos
    # K en plena metaestabilidad (donde se ve la frustracion modular), y un K
    # alto (sync global). Estos valores asumen Kc_local~0.05 y Kc_global~40,
    # acordes a la corrida anterior con N=256 / M=2 / p_intra=1 / n_aristas=1.
    K_list_temp  = [0.005, 0.15, 0.5, 2.5, 10.0, 20.0, 55.0, 100.0]
    t_max_temp   = 500.0            # u.t. — necesitamos tiempo para ver oscilaciones
    seed_temp    = 42               # ICs y A reproducibles
    method_temp  = 'rk4'            # integrador para el modo TEMP
    desfase_inducido = np.pi/100    # Desfase inducido entre los modulos

    return (N, num_modules, sigma, dt, n_aristas, p_intra, num_K, K_min, K_max, K_center, width_factor,t_max_base, t_max_peak, num_runs, methods, n_jobs, FRUSTRACION_SYNC, FRUSTRACION_TEMP, K_list_temp, t_max_temp, seed_temp, desfase_inducido, method_temp)

def setup_generico(N, num_modules, sigma, dt, n_aristas, p_intra, num_K, K_min, K_max, K_center, width_factor, t_max_base, t_max_peak, num_runs, methods, n_jobs, FRUSTRACION_SYNC, FRUSTRACION_TEMP, K_list_temp, t_max_temp, seed_temp, method_temp, run_dir):
    """
    Setup comun: imprime cabecera, guarda params.txt y devuelve la rejilla
    de K y t_max(K) para el modo SYNC. Los parametros de TEMP se registran
    en params.txt aunque SYNC se use; asi un solo params.txt documenta
    enteramente la ejecucion.
    """
    print(f"Carpeta de resultados: {run_dir}")
    print(f"Log de salida:         {os.path.join(run_dir, 'log.txt')}\n")

    guardar_params_txt(run_dir, {
        'N': N, 'num_modules': num_modules,
        'n_aristas': n_aristas, 'p_intra': p_intra,
        'sigma': sigma, 'dt': dt,
        'num_K': num_K, 'K_min': K_min, 'K_max': K_max,
        'K_center': K_center, 'width_factor': width_factor,
        't_max_base': t_max_base, 't_max_peak': t_max_peak,
        'num_runs': num_runs,
        'methods': methods,
        'n_jobs': n_jobs,
        'distribucion_K': 'log-tstudent' if K_center else 'logspace',
        'ICs_pareadas': True,
        'A_pareada_por_run': True,
        'Frustracion_Sync': FRUSTRACION_SYNC,
        'Frustracion_Temp': FRUSTRACION_TEMP,
        'K_list_temp': K_list_temp,
        't_max_temp': t_max_temp,
        'seed_temp': seed_temp,
        'method_temp': method_temp,
    })

    K_values = K_values_log_tstudent(num_K, K_min, K_max, K_center, width_factor, df=2)
    T_per_K  = t_max_per_K_log(K_values, K_center, t_max_base, t_max_peak, width_factor, df=2)
    print(f"K en [{K_values.min():.3g}, {K_values.max():.3g}]({'log-tstudent' if K_center else 'logspace'})")
    print(f"t_max(K) en [{T_per_K.min():.0f}, {T_per_K.max():.0f}] u.t.\n")

    return K_values, T_per_K

def generar_ICs_y_As_modo_sync(num_runs, N, sigma, num_modules, n_aristas, p_intra, sync_dir):
    """ICs y matrices A para el modo SYNC: una por run, pareadas con K."""
    print("Generando ICs y matrices de adyacencia (una por run)...")
    omegas_IC, thetas_IC = generar_ICs(num_runs, N, sigma)
    As, module_ids       = generar_As(num_runs, N, num_modules, n_aristas, p_intra)
    ruta_ic = guardar_ICs_y_As(omegas_IC, thetas_IC, As, module_ids, sync_dir)
    print(f"  Guardadas en: {ruta_ic}")

    ruta_A = _ruta(sync_dir, 'A_run0.png')
    plot_matriz_adyacencia(As[0], module_ids[0], ruta_A,title=fr'Run 0 — $N={N}$, $M={num_modules}$, $p_{{intra}}={p_intra}$, $\alpha={n_aristas}$')
    print(f"  Heatmap A (run 0): {ruta_A}")
    print("  Densidades por bloque (run 0):")
    print(stats_matriz_adyacencia(As[0], module_ids[0]))
    print()

    return omegas_IC, thetas_IC, As, module_ids

def generar_ICs_y_A_fijas_modo_temp(N, sigma, num_modules, n_aristas, p_intra, seed, temp_dir, desfase_inducido):
    """ICs y A unicas y reproducibles para el modo TEMP."""
    print(f"Generando ICs y A fijas (seed={seed})...")
    rng = np.random.default_rng(seed)

    if desfase_inducido is None:
        omega = rng.normal(0.0, sigma, size=N)
    else:
        sigma_mod = 0.05
        omega = np.zeros(N)
        for i in range(num_modules):
            lo = i * (N // num_modules)
            hi = (i + 1) * (N // num_modules) if i < num_modules - 1 else N
        omega[lo:hi] = rng.normal(i * desfase_inducido, sigma_mod, size=hi - lo)
    
    theta_0 = rng.uniform(-np.pi, np.pi, size=N)
    A, module_id = crear_matriz_adyacencia(N, num_modules, n_aristas, p_intra, rng)

    ruta = _ruta(temp_dir, 'condiciones_iniciales_temp.npz')
    np.savez_compressed(ruta, omega=omega, theta_0=theta_0, A=A, module_id=module_id, seed=seed)
    print(f"  Guardadas en: {ruta}")

    ruta_A = _ruta(temp_dir, 'A_temp.png')
    plot_matriz_adyacencia(A, module_id, ruta_A,title=fr'TEMP (seed={seed}) — $N={N}$, $M={num_modules}$, $p_{{intra}}={p_intra}$, $\alpha={n_aristas}$')
    print(f"  Heatmap A: {ruta_A}")
    print("  Densidades por bloque:")
    print(stats_matriz_adyacencia(A, module_id))
    print()

    return omega, theta_0, A, module_id

# ----------------------------------------------------------------------------
# Modo SYNC: barrido completo
# ----------------------------------------------------------------------------

def ejecutar_un_barrido(metodo, N, num_modules, K_values, T_per_K, num_runs, sigma, dt, omegas_IC, thetas_IC, As, module_ids, n_jobs, suffix, sync_dir):
    """Ejecuta el barrido completo para un solo metodo, guarda npz y plots,
    imprime tabla resumen y devuelve los datos crudos para combinarlos
    luego con otros metodos."""
    print(f"\n{'='*60}")
    print(f"  BARRIDO con metodo: {metodo.upper()}")
    print('='*60)
    t_metodo = time.perf_counter()

    datos = barrido_completo(N, num_modules, K_values, T_per_K, num_runs, sigma, dt, omegas_IC, thetas_IC, As, module_ids, method=metodo, n_jobs=n_jobs)
    print(f"  Tiempo barrido [{metodo}]: {(time.perf_counter()-t_metodo)/60:.1f} min")

    R_means, R_stds, R_mean_stds, rm_means, rm_stds, rm_mean_stds = datos

    fits = ajustar_todas_las_curvas(
        K_values, R_means, R_mean_stds, R_stds,
        rm_means, rm_mean_stds, rm_stds,
        num_modules, N,
    )

    np.savez_compressed(
        os.path.join(sync_dir, f'barrido{suffix}.npz'),
        K_values=K_values, T_per_K=T_per_K,
        R_means=R_means, R_stds=R_stds, R_mean_stds=R_mean_stds,
        rm_means=rm_means, rm_stds=rm_stds, rm_mean_stds=rm_mean_stds,
        num_modules=num_modules, method=metodo,
        # Guardamos los fits en el mismo archivo
        fit_global_popt=fits['global']['popt'],
        fit_global_perr=fits['global']['perr'],
        fit_modulos_popt=np.array([fits[f'm{m}']['popt'] for m in range(num_modules)]),
        fit_modulos_perr=np.array([fits[f'm{m}']['perr'] for m in range(num_modules)]),
    )

    plot_R_vs_K(K_values, R_means, R_mean_stds, rm_means, rm_mean_stds, num_modules, N, num_runs, sync_dir, suffix=suffix, fits=fits)
    plot_sigmaR_vs_K(K_values, R_stds, rm_stds, num_modules, N, num_runs, sync_dir, suffix=suffix)
    plot_combined(K_values, R_means, R_stds, R_mean_stds, rm_means, rm_stds, rm_mean_stds, num_modules, N, num_runs, sync_dir, suffix=suffix, fits=fits)

    Kc_g = Kc_experimental(K_values, R_stds, log=True)
    print("\n  " + "-" * 60)
    print(f"  [{metodo.upper()}]")
    print(f"  {'Modulo':>8} | {'Kc experimental':>16} | {'<sigma_r>':>12}")
    print(f"  {'global':>8} | {Kc_g:>16.4g} | {np.mean(R_stds):>12.5f}")
    for m in range(num_modules):
        Kc_m = Kc_experimental(K_values, rm_stds[:, m], log=True)
        print(f"  {m+1:>8} | {Kc_m:>16.4g} | {np.mean(rm_stds[:, m]):>12.5f}")

    return datos

def ejecutar_modo_sync(N, num_modules, sigma, dt, n_aristas, p_intra, K_values, T_per_K, num_runs, methods, n_jobs, run_dir):
    """Modo SYNC: barrido K, multiples runs, comparacion de metodos."""
    sync_dir = _ruta(run_dir, 'sync')
    os.makedirs(sync_dir, exist_ok=True)
    print("\n" + "#" * 60)
    print("#  MODO SYNC: barrido completo en K")
    print("#" * 60)

    omegas_IC, thetas_IC, As, module_ids = generar_ICs_y_As_modo_sync(num_runs, N, sigma, num_modules, n_aristas, p_intra, sync_dir)

    resultados_por_metodo = {}
    for metodo in methods:
        suffix = f'_{metodo}' if len(methods) > 1 else ''
        datos = ejecutar_un_barrido(metodo, N, num_modules, K_values, T_per_K, num_runs, sigma, dt, omegas_IC, thetas_IC, As, module_ids, n_jobs, suffix, sync_dir)
        resultados_por_metodo[metodo] = datos

    if len(methods) > 1:
        plot_comparacion_metodos(K_values, resultados_por_metodo, num_modules, N, num_runs, sync_dir)
        print(f"\nPlot comparativo: {os.path.join(sync_dir, 'comparacion_metodos.png')}")

# ----------------------------------------------------------------------------
# Modo TEMP: evolucion temporal con ICs fijas
# ----------------------------------------------------------------------------

def simular_evolucion_temporal(K_list, N, num_modules, sigma, dt, t_max, A, module_id, omega, theta_0, desfase_inducido, method='euler'):
    """Para cada K en K_list, ejecuta una simulacion con ICs fijas y devuelve
    una lista de dicts con las series temporales completas R(t), r_m(t)."""
    series_por_K = []
    print(f"Modo TEMP: simulando {len(K_list)} valores de K con metodo {method.upper()}...")
    for i, K in enumerate(K_list):
        t0 = time.perf_counter()
        sys = Simulacion_Kuramoto(N, num_modules, K, sigma, dt, t_max, A=A, module_id=module_id, omega=omega, theta_0=theta_0, method=method)
        elapsed = time.perf_counter() - t0
        n_pts = len(sys.R)
        R_mean_st = float(np.mean(sys.R[n_pts // 4:]))
        R_std_st  = float(np.std(sys.R[n_pts // 4:]))
        print(f"  [{i+1}/{len(K_list)}] K={K:.3g}  <R>={R_mean_st:.3f}  sigma_R={R_std_st:.3f}  ({elapsed:.1f}s)")
        series_por_K.append({'R'  : sys.R.copy(),'r_m': sys.r_m.copy(),})
    return series_por_K

def ejecutar_modo_temp(N, num_modules, sigma, dt, n_aristas, p_intra, K_list_temp, t_max_temp, seed_temp, desfase_inducido, method_temp, run_dir):
    """Modo TEMP: ICs fijas, lista de K, evolucion temporal."""
    temp_dir = _ruta(run_dir, 'temporal')
    os.makedirs(temp_dir, exist_ok=True)
    print("\n" + "#" * 60)
    print("#  MODO TEMP: evolucion temporal con ICs fijas")
    print("#" * 60)

    omega, theta_0, A, module_id = generar_ICs_y_A_fijas_modo_temp(N, sigma, num_modules, n_aristas, p_intra, seed_temp, temp_dir, desfase_inducido)

    t_inicio = time.perf_counter()
    series_por_K = simular_evolucion_temporal(K_list_temp, N, num_modules, sigma, dt, t_max_temp, A, module_id, omega, theta_0, desfase_inducido, method=method_temp)
    print(f"  Tiempo total simulaciones TEMP: {(time.perf_counter()-t_inicio)/60:.2f} min")

    # Guardar series para poder re-plotear sin re-simular
    n_pts = len(series_por_K[0]['R'])
    R_all   = np.zeros((len(K_list_temp), n_pts), dtype=np.float64)
    r_m_all = np.zeros((len(K_list_temp), num_modules, n_pts), dtype=np.float64)
    for i, s in enumerate(series_por_K):
        R_all[i]   = s['R']
        r_m_all[i] = s['r_m']
    npz_path = _ruta(temp_dir, 'series_temporales.npz')
    np.savez_compressed(npz_path, K_list=np.asarray(K_list_temp), dt=dt, t_max=t_max_temp, R=R_all, r_m=r_m_all, num_modules=num_modules, seed=seed_temp, method=method_temp)
    print(f"  Series guardadas: {npz_path}")

    # Plot apilado
    plot_evolucion_temporal(K_list_temp, series_por_K, num_modules, dt, N, temp_dir, method_name=method_temp.upper())
    print(f"  Plot: {os.path.join(temp_dir, 'evolucion_temporal.png')}")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()

    # Parametros
    (N, num_modules, sigma, dt, n_aristas, p_intra, num_K, K_min, K_max, K_center, width_factor, t_max_base, t_max_peak, num_runs, methods, n_jobs, FRUSTRACION_SYNC, FRUSTRACION_TEMP, K_list_temp, t_max_temp, seed_temp, desfase_inducido, method_temp) = parametros_simulacion()

    if not (FRUSTRACION_SYNC or FRUSTRACION_TEMP):
        print("AVISO: ni FRUSTRACION_SYNC ni FRUSTRACION_TEMP estan activados. No hay nada que ejecutar.")
        return

    # Carpeta y log
    t_max_label = f"{int(t_max_base)}-{int(t_max_peak)}"
    run_dir = crear_carpeta_resultados(N, num_modules, num_K, num_runs, t_max_label, desfase_inducido)
    log_file, stdout_orig, stderr_orig = iniciar_log(run_dir)

    try:
        # Setup comun (siempre): imprime cabecera, guarda params.txt y
        # devuelve K_values y T_per_K (utilizados solo si SYNC esta activo).
        K_values, T_per_K = setup_generico(N, num_modules, sigma, dt, n_aristas, p_intra, num_K, K_min, K_max, K_center, width_factor, t_max_base, t_max_peak, num_runs, methods, n_jobs, FRUSTRACION_SYNC, FRUSTRACION_TEMP, K_list_temp, t_max_temp, seed_temp, method_temp, run_dir)

        # Despacho de modos
        if FRUSTRACION_SYNC:
            ejecutar_modo_sync(N, num_modules, sigma, dt, n_aristas, p_intra, K_values, T_per_K, num_runs, methods, n_jobs, run_dir)

        

        if FRUSTRACION_TEMP:
            ejecutar_modo_temp(N, num_modules, sigma, dt, n_aristas, p_intra, K_list_temp, t_max_temp, seed_temp, desfase_inducido, method_temp, run_dir)

        elapsed = time.perf_counter() - t0
        print(f"\nTiempo total de ejecucion: {elapsed/60:.1f} min ({elapsed:.1f} s)")
        print(f"\nListo. Resultados en: {run_dir}")

    finally:
        cerrar_log(log_file, stdout_orig, stderr_orig)

if __name__ == "__main__":
    main()

# To do:
#
# - Ajustar parámetros para que no se corte el análisis de R.
# - Explicar por qué el RK4 es una mierda en términos de rendimiento. 
# - Repasar lo que falta de la tarea dos, dejar la primera versión terminada.