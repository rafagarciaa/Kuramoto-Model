"""
kuramoto_scripts.networks
==========================

Generadores de matrices de adyacencia sinteticas y de condiciones
iniciales pareadas (Common Random Numbers, CRN).

Contenido:
    - generar_ICs           : ICs (omega, theta_0) para n_runs.
    - generar_ICs_por_sigma : ICs por (sigma, run) para el barrido de sigma.
    - crear_matriz_modular     : red modular de 1 nivel (sim_type 1).
    - crear_matriz_jerarquica  : red modular de 2 niveles (sim_type 2).
    - generar_As_modular / generar_As_jerarquica : una matriz por run.
    - stats_matriz_adyacencia  : densidad de aristas por bloque (diagnostico).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Condiciones iniciales pareadas (Common Random Numbers)
# ----------------------------------------------------------------------------

def generar_ICs(n_runs, N, sigma, seed=None):
    """(omega, theta_0) pareadas para n_runs simulaciones.

    Devuelve omegas_IC (n_runs, N), thetas_IC (n_runs, N).
    """
    rng = np.random.default_rng(seed)
    omegas_IC = rng.normal(0.0, sigma, size=(n_runs, N))
    thetas_IC = rng.uniform(-np.pi, np.pi, size=(n_runs, N))
    return omegas_IC, thetas_IC


def generar_ICs_por_sigma(n_sigmas, n_runs, N, sigmas, seed=None):
    """ICs por (sigma, run) para el barrido en sigma (sim_type 0).

    Devuelve omegas_IC (n_sigmas, n_runs, N), thetas_IC (n_sigmas, n_runs, N).
    """
    rng = np.random.default_rng(seed)
    omegas_IC = np.zeros((n_sigmas, n_runs, N), dtype=np.float64)
    thetas_IC = np.zeros((n_sigmas, n_runs, N), dtype=np.float64)
    for i, sigma in enumerate(sigmas):
        omegas_IC[i] = rng.normal(0.0, sigma, size=(n_runs, N))
        thetas_IC[i] = rng.uniform(-np.pi, np.pi, size=(n_runs, N))
    return omegas_IC, thetas_IC


def generar_ICs_por_grupos(n_runs, N, group_id, omega_means, omega_intra_sigma,
                            seed=None):
    """ICs con frecuencias intrinsecas distintas por grupo.

    Cada oscilador i se asigna a un grupo g = group_id[i]. Su omega_i se
    muestrea de N(omega_means[g], omega_intra_sigma^2). Las theta_0 siguen
    siendo uniformes en [-pi, pi].

    Util para simular el cerebro: distintas regiones tienen frecuencias
    caracteristicas (por hemisferio, por lobulo, por banda fisiologica).

    Parametros
    ----------
    n_runs : int
    N : int
    group_id : (N,) int
        Indice de grupo de cada oscilador. Valores 0..G-1.
    omega_means : (G,) float
        Frecuencia media de cada grupo.
    omega_intra_sigma : float
        Dispersion dentro de cada grupo (la misma para todos).

    Devuelve
    --------
    omegas_IC : (n_runs, N) float
    thetas_IC : (n_runs, N) float
    """
    rng = np.random.default_rng(seed)
    group_id = np.asarray(group_id, dtype=np.int64)
    omega_means = np.asarray(omega_means, dtype=np.float64)
    G = int(group_id.max()) + 1
    if len(omega_means) != G:
        raise ValueError(f"omega_means tiene {len(omega_means)} valores pero "
                         f"hay {G} grupos en group_id.")

    omegas_IC = np.zeros((n_runs, N), dtype=np.float64)
    for r in range(n_runs):
        for i in range(N):
            g = int(group_id[i])
            omegas_IC[r, i] = rng.normal(omega_means[g], omega_intra_sigma)

    thetas_IC = rng.uniform(-np.pi, np.pi, size=(n_runs, N))
    return omegas_IC, thetas_IC


# ----------------------------------------------------------------------------
# Helper: añadir aristas aleatorias distintas entre dos conjuntos de nodos
# ----------------------------------------------------------------------------

def _add_random_edges(A, nodes_a, nodes_b, n_edges, rng):
    """Añade n_edges aristas distintas entre nodes_a y nodes_b (in-place).

    nodes_a y nodes_b son arrays de indices de nodo. Para conexiones
    intra-grupo basta pasar el mismo array dos veces (se evita el bucle
    propio y los duplicados).
    """
    añadidas = 0
    intentos = 0
    max_intentos = 1000 * max(n_edges, 1)
    while añadidas < n_edges and intentos < max_intentos:
        intentos += 1
        a = int(rng.choice(nodes_a))
        b = int(rng.choice(nodes_b))
        if a == b or A[a, b] == 1:
            continue
        A[a, b] = 1.0
        A[b, a] = 1.0
        añadidas += 1


# ----------------------------------------------------------------------------
# Red modular de 1 nivel (sim_type 1)
# ----------------------------------------------------------------------------

def crear_matriz_modular(N, n_modules, n_edges_inter, p_intra, rng=None):
    """Red modular simetrica: n_modules bloques densos, debilmente unidos.

        - Cada bloque: N // n_modules nodos (el ultimo absorbe el resto).
        - Intra-bloque: arista con probabilidad p_intra.
        - Inter-bloques: n_edges_inter aristas por cada par de modulos.

    Devuelve A (N,N) float64 simetrica diagonal 0, y module_id (N,).
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((N, N), dtype=np.float64)
    module_id = np.zeros(N, dtype=np.int64)

    tam = N // n_modules
    for m in range(n_modules - 1):
        module_id[m * tam:(m + 1) * tam] = m
    module_id[(n_modules - 1) * tam:] = n_modules - 1

    # Intra-bloque.
    for i in range(N):
        for j in range(i + 1, N):
            if module_id[i] == module_id[j] and rng.random() < p_intra:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Inter-bloques.
    for m1 in range(n_modules):
        nodos_1 = np.where(module_id == m1)[0]
        for m2 in range(m1 + 1, n_modules):
            nodos_2 = np.where(module_id == m2)[0]
            _add_random_edges(A, nodos_1, nodos_2, n_edges_inter, rng)

    return A, module_id


def generar_As_modular(n_runs, N, n_modules, n_edges_inter, p_intra, seed=None):
    """n_runs matrices modulares independientes. module_id es comun (mismo layout).

    Devuelve As (n_runs, N, N), module_id (N,).
    """
    rng = np.random.default_rng(seed)
    As = np.zeros((n_runs, N, N), dtype=np.float64)
    module_id = None
    for r in range(n_runs):
        A, mid = crear_matriz_modular(N, n_modules, n_edges_inter, p_intra, rng)
        As[r] = A
        module_id = mid
    return As, module_id


# ----------------------------------------------------------------------------
# Red jerarquica de 2 niveles (sim_type 2)
# ----------------------------------------------------------------------------

def crear_matriz_jerarquica(N, submodules_per_module,
                            p_intra_submodule,
                            n_edges_inter_submodule,
                            n_edges_inter_module,
                            rng=None):
    """Red jerarquica de 2 niveles.

    Estructura (todo simetrico):
        - submodules_per_module[m] = nº de submodulos dentro del modulo m.
        - Total submodulos = sum(submodules_per_module). Todos del mismo
          tamaño (el ultimo absorbe el resto si N no es divisible).
        - Intra-submodulo: arista con probabilidad p_intra_submodule (denso).
        - Inter-submodulos del MISMO modulo: n_edges_inter_submodule aristas
          por cada par de submodulos.
        - Inter-modulos: n_edges_inter_module aristas por cada par de modulos
          (los menos: cuello de botella jerarquico).

    Devuelve
    --------
    A            : (N, N) float64
    module_id    : (N,) int64  -> modulo de nivel 1 de cada nodo.
    submodule_id : (N,) int64  -> submodulo de nivel 2 (indice GLOBAL de submodulo).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_modules = len(submodules_per_module)

    module_id    = np.zeros(N, dtype=np.int64)
    submodule_id = np.zeros(N, dtype=np.int64)

    # Reparto en cascada en DOS niveles para que los modulos sean del
    # mismo tamano y, DENTRO de cada modulo, sus submodulos tambien lo sean:
    #
    #   1) N se reparte equitativamente entre los n_modules.
    #      El ultimo modulo absorbe el resto si N no es divisible.
    #   2) Dentro de cada modulo, sus nodos se reparten equitativamente
    #      entre sus submodules_per_module[m] submodulos.
    #      El ultimo submodulo del modulo absorbe su resto.
    #
    # Asi, modulos con distinto numero de submodulos seguiran teniendo el
    # mismo numero de nodos; lo que cambia es el tamano interno de cada
    # submodulo.

    tam_modulo = N // n_modules
    sub_global = 0
    nodo = 0
    for m in range(n_modules):
        # Nodos asignados a este modulo (el ultimo absorbe el resto).
        es_ultimo_modulo = (m == n_modules - 1)
        fin_modulo = N if es_ultimo_modulo else nodo + tam_modulo
        nodos_modulo = fin_modulo - nodo
        module_id[nodo:fin_modulo] = m

        # Dentro del modulo, reparto entre sus submodulos.
        n_subs = submodules_per_module[m]
        tam_sub = nodos_modulo // n_subs
        nodo_sub = nodo
        for s in range(n_subs):
            es_ultimo_sub = (s == n_subs - 1)
            fin_sub = fin_modulo if es_ultimo_sub else nodo_sub + tam_sub
            submodule_id[nodo_sub:fin_sub] = sub_global
            nodo_sub = fin_sub
            sub_global += 1

        nodo = fin_modulo

    # Nivel 2: intra-submodulo denso.
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            if submodule_id[i] == submodule_id[j] and rng.random() < p_intra_submodule:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Inter-submodulos dentro del mismo modulo.
    for m in range(n_modules):
        subs_m = np.unique(submodule_id[module_id == m])
        for a in range(len(subs_m)):
            nodos_a = np.where(submodule_id == subs_m[a])[0]
            for b in range(a + 1, len(subs_m)):
                nodos_b = np.where(submodule_id == subs_m[b])[0]
                _add_random_edges(A, nodos_a, nodos_b, n_edges_inter_submodule, rng)

    # Inter-modulos (el nivel mas debil).
    for m1 in range(n_modules):
        nodos_1 = np.where(module_id == m1)[0]
        for m2 in range(m1 + 1, n_modules):
            nodos_2 = np.where(module_id == m2)[0]
            _add_random_edges(A, nodos_1, nodos_2, n_edges_inter_module, rng)

    return A, module_id, submodule_id


def generar_As_jerarquica(n_runs, N, submodules_per_module,
                          p_intra_submodule, n_edges_inter_submodule,
                          n_edges_inter_module, seed=None):
    """n_runs matrices jerarquicas. module_id/submodule_id comunes (mismo layout).

    Devuelve As (n_runs, N, N), module_id (N,), submodule_id (N,).
    """
    rng = np.random.default_rng(seed)
    As = np.zeros((n_runs, N, N), dtype=np.float64)
    module_id = submodule_id = None
    for r in range(n_runs):
        A, mid, sid = crear_matriz_jerarquica(
            N, submodules_per_module, p_intra_submodule,
            n_edges_inter_submodule, n_edges_inter_module, rng)
        As[r] = A
        module_id, submodule_id = mid, sid
    return As, module_id, submodule_id


# ----------------------------------------------------------------------------
# Diagnostico
# ----------------------------------------------------------------------------

def stats_matriz_adyacencia(A, group_id):
    """Densidad de aristas por bloque (intra/inter grupo). (n_grupos, n_grupos)."""
    n_grupos = int(group_id.max()) + 1
    densidades = np.zeros((n_grupos, n_grupos))
    for i in range(n_grupos):
        idx_i = np.where(group_id == i)[0]
        for j in range(n_grupos):
            idx_j = np.where(group_id == j)[0]
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
