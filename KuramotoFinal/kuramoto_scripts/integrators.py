"""
kuramoto_scripts.integrators
=============================

Integradores Euler explicitos compilados con Numba (njit), CON PARADA
ADAPTATIVA POR CONVERGENCIA.

    _euler_meanfield  : campo medio, coste O(N) por paso (sim_type 0).
    _euler_network    : red arbitraria, coste O(N^2) por paso, calcula
                        R global + el parametro de orden r de cada grupo
                        en uno o varios niveles (sim_type 1, 2, 3).

Parada por convergencia (sustituye al antiguo t_max fijo por punto):

    Cada `block_size` pasos promediamos los R de ese bloque y lo guardamos.
    Cuando dos medias de bloques CONSECUTIVOS difieren menos que
    `conv_threshold`, consideramos que R(t) se ha estabilizado y paramos.
    `max_steps` es un tope de seguridad por si nunca converge (p.ej. justo
    en Kc, donde R fluctua de forma sostenida).

    El observable ⟨R⟩ y su fluctuacion sigma_R de la simulacion se calculan
    luego (en system.py) sobre los DOS ULTIMOS bloques (2*block_size valores).

Las funciones rellenan R, psi (y r_levels) IN PLACE hasta el paso de
parada y devuelven `n_steps_used` (ultimo indice de tiempo calculado).
"""

import math
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# Integrador 1: campo medio  (sim_type 0)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _euler_meanfield(theta, theta_new, omega, K, dt,
                     max_steps, block_size, conv_threshold,
                     R, psi):
    """Euler campo medio con parada por convergencia.

        theta_dot_i = omega_i + K * R(t) * sin(psi(t) - theta_i)

    Devuelve n_steps_used (indice del ultimo R calculado).
    """
    N = theta.shape[0]

    acc        = 0.0     # acumulador del bloque actual
    block_prev = 0.0     # media del bloque anterior
    tiene_prev = False   # ¿ya hay un bloque anterior con el que comparar?
    n_used     = max_steps - 1

    for t in range(max_steps):
        # --- Observable global R(t), psi(t) ---
        re, im = 0.0, 0.0
        for j in range(N):
            re += math.cos(theta[j])
            im += math.sin(theta[j])
        R[t]   = math.sqrt(re*re + im*im) / N
        psi[t] = math.atan2(im, re)

        acc += R[t]

        # --- Chequeo de convergencia en frontera de bloque ---
        if (t + 1) % block_size == 0:
            block_mean = acc / block_size
            acc = 0.0
            if tiene_prev and abs(block_mean - block_prev) < conv_threshold:
                n_used = t
                break
            block_prev = block_mean
            tiene_prev = True

        # --- Paso de Euler ---
        for i in range(N):
            theta_new[i] = theta[i] + dt * (
                omega[i] + K * R[t] * math.sin(psi[t] - theta[i])
            )
        for i in range(N):
            theta[i] = theta_new[i]

    return n_used


# ----------------------------------------------------------------------------
# Integrador 2: red arbitraria con niveles  (sim_type 1, 2, 3)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _euler_network(theta, theta_new, omega, A, K, dt,
                   max_steps, block_size, conv_threshold,
                   level_id, n_groups_per_level, group_size,
                   R, psi, r_levels, rhs):
    """Euler en red con parada por convergencia y r de orden por nivel.

        theta_dot_i = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)

    Parametros de niveles
    ---------------------
    level_id : int (L, N)
        level_id[l, i] = indice de grupo del nodo i en el nivel l.
        Ej.: nivel 0 = modulos, nivel 1 = submodulos.
    n_groups_per_level : int (L,)
        Numero de grupos en cada nivel.
    group_size : float (L, Gmax)
        Tamaño (nº de nodos) de cada grupo. Gmax = max sobre niveles.
    r_levels : float (L, Gmax, max_steps+1)
        Salida: parametro de orden de cada grupo y nivel en el tiempo.

    El parametro de orden de un grupo g se calcula con la suma compleja de
    SOLO los nodos de ese grupo: r_g = |sum_{j in g} e^{i theta_j}| / |g|.
    Esto permite r de submodulo (nivel fino) y r de modulo (nivel grueso)
    de forma independiente y exacta en cada nivel.

    Devuelve n_steps_used.
    """
    N = theta.shape[0]
    L = level_id.shape[0]
    Gmax = r_levels.shape[1]

    # Buffers de las sumas complejas por (nivel, grupo).
    re_g = np.zeros((L, Gmax), dtype=np.float64)
    im_g = np.zeros((L, Gmax), dtype=np.float64)

    acc        = 0.0
    block_prev = 0.0
    tiene_prev = False
    n_used     = max_steps - 1

    for t in range(max_steps):
        # --- Observables: R global + r por nivel/grupo, en un solo pase ---
        re_tot, im_tot = 0.0, 0.0
        for l in range(L):
            for g in range(n_groups_per_level[l]):
                re_g[l, g] = 0.0
                im_g[l, g] = 0.0

        for j in range(N):
            cj = math.cos(theta[j])
            sj = math.sin(theta[j])
            re_tot += cj
            im_tot += sj
            for l in range(L):
                g = level_id[l, j]
                re_g[l, g] += cj
                im_g[l, g] += sj

        R[t]   = math.sqrt(re_tot*re_tot + im_tot*im_tot) / N
        psi[t] = math.atan2(im_tot, re_tot)
        for l in range(L):
            for g in range(n_groups_per_level[l]):
                mag = math.sqrt(re_g[l, g]*re_g[l, g] + im_g[l, g]*im_g[l, g])
                r_levels[l, g, t] = mag / group_size[l, g]

        acc += R[t]

        # --- Chequeo de convergencia ---
        if (t + 1) % block_size == 0:
            block_mean = acc / block_size
            acc = 0.0
            if tiene_prev and abs(block_mean - block_prev) < conv_threshold:
                n_used = t
                break
            block_prev = block_mean
            tiene_prev = True

        # --- rhs O(N^2) y paso de Euler ---
        for i in range(N):
            ti = theta[i]
            c  = 0.0
            for j in range(N):
                c += A[i, j] * math.sin(theta[j] - ti)
            rhs[i] = omega[i] + K * c
        for i in range(N):
            theta_new[i] = theta[i] + dt * rhs[i]
        for i in range(N):
            theta[i] = theta_new[i]

    return n_used


# ----------------------------------------------------------------------------
# Integrador 3: red sin parada por convergencia, con historico de theta
# (sim_type 5: dinamica cerebral, snapshots, MP4)
# ----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _euler_network_long(theta, theta_new, omega, A, K, dt, n_steps,
                        level_id, n_groups_per_level, group_size,
                        R, psi, r_levels, rhs,
                        theta_history, sample_every):
    """Euler en red SIN parada por convergencia, con historico de theta.

    Diferencias clave respecto a _euler_network:
      - Corre EXACTAMENTE n_steps pasos (no hay early stop).
      - Cada `sample_every` pasos guarda theta[:] en theta_history[k, :].
        Esto permite hacer snapshots de fase y animaciones a posteriori.

    Restricciones:
      - n_steps debe ser <= len(R) y len(R) -1 (mismo contrato que _euler_network).
      - theta_history.shape = (n_samples, N) con n_samples = ceil(n_steps/sample_every).

    Devuelve numero efectivo de muestras guardadas en theta_history.
    """
    N = theta.shape[0]
    L = level_id.shape[0]
    Gmax = r_levels.shape[1]

    re_g = np.zeros((L, Gmax), dtype=np.float64)
    im_g = np.zeros((L, Gmax), dtype=np.float64)

    n_saved = 0

    for t in range(n_steps):
        # --- Observables: R global + r por nivel/grupo en un solo pase ---
        re_tot, im_tot = 0.0, 0.0
        for l in range(L):
            for g in range(n_groups_per_level[l]):
                re_g[l, g] = 0.0
                im_g[l, g] = 0.0

        for j in range(N):
            cj = math.cos(theta[j])
            sj = math.sin(theta[j])
            re_tot += cj
            im_tot += sj
            for l in range(L):
                g = level_id[l, j]
                re_g[l, g] += cj
                im_g[l, g] += sj

        R[t]   = math.sqrt(re_tot*re_tot + im_tot*im_tot) / N
        psi[t] = math.atan2(im_tot, re_tot)
        for l in range(L):
            for g in range(n_groups_per_level[l]):
                mag = math.sqrt(re_g[l, g]*re_g[l, g] + im_g[l, g]*im_g[l, g])
                r_levels[l, g, t] = mag / group_size[l, g]

        # --- Guardar theta en el historico (cada sample_every pasos) ---
        if t % sample_every == 0 and n_saved < theta_history.shape[0]:
            for i in range(N):
                theta_history[n_saved, i] = theta[i]
            n_saved += 1

        # --- rhs O(N^2) y paso de Euler ---
        for i in range(N):
            ti = theta[i]
            c  = 0.0
            for j in range(N):
                c += A[i, j] * math.sin(theta[j] - ti)
            rhs[i] = omega[i] + K * c
        for i in range(N):
            theta_new[i] = theta[i] + dt * rhs[i]
        for i in range(N):
            theta[i] = theta_new[i]

    return n_saved
