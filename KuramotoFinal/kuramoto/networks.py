"""
kuramoto.networks
=================

Generadores de matrices de adyacencia sinteticas y de condiciones
iniciales pareadas (Common Random Numbers, CRN).

Que vive aqui:

    - generar_ICs           : ICs Gaussianas/uniformes para num_runs.
    - generar_ICs_por_sigma : ICs por (sigma, run) para barridos de sigma.
    - crear_matriz_modular  : matriz modular tipo Tarea 2 (M bloques densos
                              con n_aristas inter-modulos).
    - generar_As_modular    : varias matrices independientes para num_runs.
    - stats_matriz_adyacencia: densidad de aristas por bloque (diagnostico).

NOTA: NO hay un crear_jerarquica aqui. Para Tarea 3/4 el usuario pasa
directamente la matriz A (ya sea generada externamente o cargada con
kuramoto.conectoma). La filosofia es: si tienes la matriz, no necesitas
que el paquete te la construya.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Condiciones iniciales pareadas (Common Random Numbers)
# ----------------------------------------------------------------------------

def generar_ICs(num_runs, N, sigma, seed=None):
    """Genera (omega, theta_0) pareadas para `num_runs` simulaciones.

    Justificacion del pareado:
        Para estimar <R>(K) sobre una rejilla de K, queremos comparar la
        MISMA realizacion de IC para todos los K. Eso reduce muchisimo
        la varianza cruzada entre celdas y suaviza la curva R(K).

    Devuelve
    --------
    omegas_IC : ndarray, shape (num_runs, N)
    thetas_IC : ndarray, shape (num_runs, N)
    """
    rng = np.random.default_rng(seed)
    omegas_IC = rng.normal(0.0, sigma, size=(num_runs, N))
    thetas_IC = rng.uniform(-np.pi, np.pi, size=(num_runs, N))
    return omegas_IC, thetas_IC


def generar_ICs_por_sigma(num_sigmas, num_runs, N, sigma_values, seed=None):
    """Variante para barridos en sigma (Tarea 1): un set de ICs por sigma.

    Devuelve
    --------
    omegas_IC : ndarray, shape (num_sigmas, num_runs, N)
    thetas_IC : ndarray, shape (num_sigmas, num_runs, N)
    """
    rng = np.random.default_rng(seed)
    omegas_IC = np.zeros((num_sigmas, num_runs, N), dtype=np.float64)
    thetas_IC = np.zeros((num_sigmas, num_runs, N), dtype=np.float64)
    for i, sigma in enumerate(sigma_values):
        omegas_IC[i] = rng.normal(0.0, sigma, size=(num_runs, N))
        thetas_IC[i] = rng.uniform(-np.pi, np.pi, size=(num_runs, N))
    return omegas_IC, thetas_IC


# ----------------------------------------------------------------------------
# Matriz modular sintetica (Tarea 2)
# ----------------------------------------------------------------------------

def crear_matriz_modular(N, num_modules, n_aristas, p_intra, rng=None):
    """Construye una matriz de adyacencia modular: M bloques densos
    debilmente interconectados.

    Modelo:
        - Cada bloque tiene N // num_modules nodos (el ultimo absorbe el
          resto si N no es divisible).
        - Intra-bloque: cada par (i, j) con i < j en el mismo bloque
          recibe una arista con probabilidad p_intra.
        - Inter-bloques: para cada par de modulos (m1, m2), se sortean
          exactamente n_aristas aristas distintas entre nodos al azar
          de ambos bloques.

    Esto es lo mas parecido a "frustracion modular" del enunciado: pocos
    enlaces inter-modulares -> cada modulo se sincroniza internamente
    rapido pero la coherencia global es lenta y oscila (metaestabilidad).

    Parametros
    ----------
    N : int
    num_modules : int
    n_aristas : int
        Numero de aristas inter-modulos por cada par (m1, m2).
    p_intra : float in [0, 1]
        Probabilidad de arista intra-modulo.
    rng : np.random.Generator o None

    Devuelve
    --------
    A : ndarray (N, N), float64
        Simetrica, diagonal 0, entradas 0/1.
    module_id : ndarray (N,), int64
        El modulo (0..num_modules-1) al que pertenece cada nodo.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((N, N), dtype=np.float64)
    module_id = np.zeros(N, dtype=np.int64)

    # Asignamos los primeros num_modules-1 bloques con tamano N//num_modules.
    for m in range(num_modules - 1):
        module_id[m * (N // num_modules) : (m + 1) * (N // num_modules)] = m
    # El ultimo bloque absorbe el resto (caso N no divisible incluido).
    module_id[(num_modules - 1) * (N // num_modules):] = num_modules - 1

    # Intra-bloque: aristas con probabilidad p_intra.
    for i in range(N):
        for j in range(i + 1, N):
            if module_id[i] == module_id[j]:
                if rng.random() < p_intra:
                    A[i, j] = 1
                    A[j, i] = 1

    # Inter-bloques: sorteo exacto de n_aristas por cada par de modulos.
    for m1 in range(num_modules):
        for m2 in range(m1 + 1, num_modules):
            for _ in range(n_aristas):
                a = rng.integers(low=m1 * (N // num_modules),
                                 high=(m1 + 1) * (N // num_modules))
                b = rng.integers(low=m2 * (N // num_modules),
                                 high=(m2 + 1) * (N // num_modules))
                # Si ya existia, sigue intentandolo hasta encontrar par libre.
                while A[a, b] == 1:
                    a = rng.integers(low=m1 * (N // num_modules),
                                     high=(m1 + 1) * (N // num_modules))
                    b = rng.integers(low=m2 * (N // num_modules),
                                     high=(m2 + 1) * (N // num_modules))
                A[a, b] = 1
                A[b, a] = 1

    return A, module_id


def generar_As_modular(num_runs, N, num_modules, n_aristas, p_intra, seed=None):
    """Genera `num_runs` matrices modulares independientes con sus module_id.

    Una matriz por run, todas con los mismos parametros estructurales pero
    realizaciones distintas. Esto permite hacer CRN tambien sobre la red:
    la run r usa siempre las mismas IC y la misma A para todos los K.

    Devuelve
    --------
    As         : ndarray (num_runs, N, N)
    module_ids : ndarray (num_runs, N)
    """
    rng = np.random.default_rng(seed)
    As = np.zeros((num_runs, N, N), dtype=np.float64)
    module_ids = np.zeros((num_runs, N), dtype=np.int64)
    for r in range(num_runs):
        A, mid = crear_matriz_modular(N, num_modules, n_aristas, p_intra, rng)
        As[r]         = A
        module_ids[r] = mid
    return As, module_ids


# ----------------------------------------------------------------------------
# Diagnostico
# ----------------------------------------------------------------------------

def stats_matriz_adyacencia(A, module_id):
    """Densidad de aristas en cada bloque (i, j) de modulos.

    Devuelve una matriz (num_modules, num_modules) con valores en [0, 1]:
        diagonal -> densidad intra-modulo.
        fuera    -> densidad inter-modulos.

    Util para verificar que crear_matriz_modular hizo lo esperado.
    """
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
