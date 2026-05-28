"""
kuramoto.integrators
====================

Dos integradores Euler explicitos, ambos compilados con Numba (njit).

    _euler_meanfield   : campo medio, coste O(N) por paso.
    _euler_red         : red arbitraria, coste O(N^2) por paso.

Decision de NO incluir RK4:
    RK4 cuesta 4x mas por paso que Euler y solo aporta orden de precision
    en sistemas suaves. En Kuramoto el seno satura, asi que el bound
    practico de estabilidad lineal apenas mejora (2 -> 2.78) y para los
    barridos masivos que hacemos sale mucho mas rentable usar Euler con
    dt pequeño. Lo retiramos para mantener el codigo simple.

Las dos funciones modifican `R`, `psi` y opcionalmente `r_m` IN PLACE.
"""

import math
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# Helpers compartidos (Numba)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _obs_global(theta, R, psi, t):
    """Calcula R(t) y psi(t) globales a partir de la suma compleja
    Re^(i*psi) = (1/N) * sum_j e^(i*theta_j).

    Esta es la rama campo-medio (no necesita module_id).
    """
    N = theta.shape[0]
    re, im = 0.0, 0.0
    for j in range(N):
        re += math.cos(theta[j])
        im += math.sin(theta[j])
    R[t]   = math.sqrt(re*re + im*im) / N
    psi[t] = math.atan2(im, re)


@njit(fastmath=True, cache=True)
def _obs_global_y_modulos(theta, module_id, module_size,
                          R, psi, r_m, t, re_m, im_m):
    """Calcula R(t), psi(t) y r_m(t) por modulo en un unico pase.

    Esta es la rama de red. `module_size[m]` se precomputa fuera para
    evitar recalcularlo cada paso.

    `re_m` e `im_m` son buffers de tamaño M = num_modules que se
    reutilizan; los reseteamos a 0 al principio del calculo.
    """
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
# Integrador 1: campo medio  (Tarea 1)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _euler_meanfield(theta_curr, theta_next, omega, K, dt, steps, R, psi):
    """Euler explicito en la forma de campo medio.

    Ecuacion (equivalente a Aij = 1/N en (1)):

        theta_dot_i = omega_i + K * R(t) * sin(psi(t) - theta_i)

    Coste por paso: O(N). La gran ganancia respecto a la red completa
    es que el termino de acoplamiento se reduce a un producto escalar
    R*sin(psi-theta_i), no a una suma sobre N vecinos.

    Cota de estabilidad lineal: dt * K < 2 (aproximada).
    Precision: O(dt).
    """
    N = theta_curr.shape[0]

    for t in range(steps):
        _obs_global(theta_curr, R, psi, t)

        # Euler: usamos R[t] y psi[t] recien calculados como campo medio.
        for i in range(N):
            theta_next[i] = theta_curr[i] + dt * (
                omega[i] + K * R[t] * math.sin(psi[t] - theta_curr[i])
            )

        # Swap de buffers: theta_curr <- theta_next.
        for i in range(N):
            theta_curr[i] = theta_next[i]

    # Ultimo paso: R y psi en t = steps.
    _obs_global(theta_curr, R, psi, steps)


# ----------------------------------------------------------------------------
# Integrador 2: red arbitraria  (Tareas 2, 3, 4)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _eval_rhs_red(theta, omega, A, K, rhs):
    """Evalua el rhs de Kuramoto en red, in-place sobre `rhs`:

        rhs[i] = omega[i] + K * sum_j A[i,j] * sin(theta[j] - theta[i])

    Este es el bottleneck O(N^2) del integrador de red. Numba lo deberia
    inlinar dentro de _euler_red.
    """
    N = theta.shape[0]
    for i in range(N):
        ti = theta[i]
        c  = 0.0
        for j in range(N):
            c += A[i, j] * math.sin(theta[j] - ti)
        rhs[i] = omega[i] + K * c


@njit(fastmath=True, cache=True)
def _euler_red(theta_curr, theta_next, omega, A, module_id,
               K, dt, steps, R, psi, r_m):
    """Euler explicito en la forma de red.

    Ecuacion:

        theta_dot_i = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)

    Sin el 1/N: K es ahora el acoplamiento POR ENLACE.

    Coste por paso: O(N^2) si A es densa, dominado por _eval_rhs_red.
    Cota de estabilidad lineal: dt * K * lambda_max(L) < 2.
    Precision: O(dt).
    """
    N = theta_curr.shape[0]
    M = r_m.shape[0]

    # Precompute size of each module: vector usado en _obs_global_y_modulos.
    module_size = np.zeros(M, dtype=np.float64)
    for i in range(N):
        module_size[module_id[i]] += 1.0

    re_m = np.zeros(M, dtype=np.float64)
    im_m = np.zeros(M, dtype=np.float64)
    rhs  = np.zeros(N, dtype=np.float64)

    for t in range(steps):
        _obs_global_y_modulos(theta_curr, module_id, module_size,
                              R, psi, r_m, t, re_m, im_m)

        _eval_rhs_red(theta_curr, omega, A, K, rhs)
        for i in range(N):
            theta_next[i] = theta_curr[i] + dt * rhs[i]

        for i in range(N):
            theta_curr[i] = theta_next[i]

    _obs_global_y_modulos(theta_curr, module_id, module_size,
                          R, psi, r_m, steps, re_m, im_m)
