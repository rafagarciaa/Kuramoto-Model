"""
kuramoto_scripts.conectoma
==========================

Lectura y preparacion del conectoma estructural humano (sim_type 3).

Datos: SCmatrices88healthy.mat (Domhof et al. 2022): 88 matrices 90x90
(una por sujeto sano), conectividad estructural entre las 90 regiones AAL.

Funciones:
    cargar_conectoma_promedio / cargar_conectoma_sujetos : leen el .mat.
    binarizar(W, thr)                : pesos -> adyacencia binaria simetrica.
    es_conexo(A)                     : ¿una sola componente?
    buscar_threshold_optimo(W)       : thr mas alto que mantiene conexo.
    cargar_y_preparar_A(ruta, thr)   : pipeline .mat -> A binaria.
    hemisferio_ids(csv)              : etiqueta L/R de cada region (2 grupos).
    randomize_preserving_degree(A)   : red aleatoria con el MISMO grado
                                       (double-edge swap). Sirve de control:
                                       comparar metaestabilidad conectoma vs
                                       azar demuestra que la estructura modular
                                       del cerebro la maximiza.
"""

import os
import csv
import numpy as np
import scipy.io as sio


# ----------------------------------------------------------------------------
# Lectura del .mat
# ----------------------------------------------------------------------------

def cargar_conectoma_sujetos(ruta_mat, clave='SCmatrices'):
    """Matrices INDIVIDUALES de los 88 sujetos, shape (88, 90, 90)."""
    data = sio.loadmat(ruta_mat)
    return data[clave]


def cargar_conectoma_promedio(ruta_mat, clave='SCmatrices'):
    """Promedio sobre sujetos -> matriz de pesos W (90, 90)."""
    sujetos = cargar_conectoma_sujetos(ruta_mat, clave=clave)
    return np.mean(sujetos, axis=0)


# ----------------------------------------------------------------------------
# Binarizacion
# ----------------------------------------------------------------------------

def binarizar(W, threshold):
    """Pesos W -> adyacencia binaria A (simetrica, diagonal 0).

        A[i,j] = 1  si  W_sim[i,j] > threshold,  con  W_sim = (W + W.T)/2.

    Simetrizamos porque el conectoma es fisicamente no direccional; la
    asimetria de W es ruido de tractografia. El threshold descarta fibras
    debiles poco fiables.
    """
    W_sim = 0.5 * (W + W.T)
    A = (W_sim > threshold).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A


# ----------------------------------------------------------------------------
# Conectividad
# ----------------------------------------------------------------------------

def es_conexo(A):
    """True si el grafo de A es de una sola componente (BFS via matriz-vector).

    Importa porque en Kuramoto componentes desconectadas no pueden
    sincronizarse entre si por mucho que suba K: queremos UN cerebro.
    """
    N = A.shape[0]
    visitados = np.zeros(N, dtype=bool)
    visitados[0] = True
    while True:
        nuevos = visitados | (A.dot(visitados.astype(np.float64)) > 0)
        if np.array_equal(nuevos, visitados):
            break
        visitados = nuevos
    return bool(visitados.all())


def buscar_threshold_optimo(W, n_thresholds=300, verbose=True):
    """Threshold MAS ALTO que aun deja el grafo conexo.

    Demasiado bajo -> red casi completa (sin estructura). Demasiado alto ->
    grafo fragmentado. El punto justo limpia el maximo ruido conservando
    una sola componente.

    Devuelve (thr_optimo, info_dict).
    """
    thresholds = np.linspace(1e-6, W.max(), n_thresholds)
    densidades, conexos = [], []
    N = W.shape[0]
    n_pares = N * (N - 1) // 2

    for thr in thresholds:
        A = binarizar(W, thr)
        densidades.append(int(A.sum() // 2) / n_pares)
        conexos.append(es_conexo(A))

    densidades = np.array(densidades)
    conexos    = np.array(conexos)

    idx_conexos = np.where(conexos)[0]
    if len(idx_conexos) == 0:
        raise ValueError("Ningun threshold da grafo conexo. Revisar W.")
    idx = idx_conexos.max()
    thr_optimo = thresholds[idx]

    if verbose:
        print(f"Threshold optimo: {thr_optimo:.6f}  "
              f"(densidad {densidades[idx]:.4f}, "
              f"{int(densidades[idx]*n_pares)} enlaces)")

    return thr_optimo, {'thresholds': thresholds,
                        'densidades': densidades,
                        'conexos': conexos}


def cargar_y_preparar_A(ruta_mat, threshold='auto', clave='SCmatrices', verbose=True):
    """Pipeline: lee .mat -> W promedio -> threshold -> A binaria float64.

    threshold : 'auto'  -> usa buscar_threshold_optimo.
                float    -> se usa tal cual.

    Devuelve (A, thr_usado).
    """
    W = cargar_conectoma_promedio(ruta_mat, clave=clave)

    if isinstance(threshold, str) and threshold == 'auto':
        thr, _ = buscar_threshold_optimo(W, verbose=verbose)
    else:
        thr = float(threshold)
        if verbose:
            print(f"Threshold explicito: {thr:.6f}")

    A = binarizar(W, thr)

    if verbose:
        N = A.shape[0]
        n_pares = N * (N - 1) // 2
        n_enlaces = int(A.sum() // 2)
        print(f"Conectoma preparado: N={N}, |E|={n_enlaces}, "
              f"densidad={n_enlaces/n_pares:.4f}")

    return A, thr


# ----------------------------------------------------------------------------
# Hemisferios (niveles para r_m)
# ----------------------------------------------------------------------------

def hemisferio_ids(ruta_csv):
    """Lee AAL_regions.csv y devuelve un array (90,) con 0=izquierdo, 1=derecho.

    El CSV tiene formato 'ROI number;ROI name' y los nombres empiezan por
    'L ' (left) o 'R ' (right). Usamos ese prefijo para asignar hemisferio.
    """
    hemis = []
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        lector = csv.reader(f, delimiter=';')
        next(lector, None)  # cabecera
        for fila in lector:
            if len(fila) < 2:
                continue
            nombre = fila[1].strip()
            hemis.append(0 if nombre.upper().startswith('L') else 1)
    return np.asarray(hemis, dtype=np.int64)


# ----------------------------------------------------------------------------
# Red aleatoria que preserva el grado (control de metaestabilidad)
# ----------------------------------------------------------------------------

def randomize_preserving_degree(A, n_swaps_factor=10, rng=None):
    """Devuelve una red aleatoria con EXACTAMENTE el mismo grado que A.

    Metodo: double-edge swap (Maslov-Sneppen). Repetidamente toma dos
    aristas (a,b) y (c,d) y las recablea a (a,d) y (c,b) si no crea
    bucles ni duplicados. Cada nodo conserva su grado, pero la estructura
    modular se destruye. Numero de intercambios = n_swaps_factor * |E|.

    Comparar la metaestabilidad de A (conectoma) con la de esta version
    aleatoria aisla el efecto de la ESTRUCTURA: si sigma_R es mayor en el
    conectoma, la modularidad jerarquica del cerebro maximiza la
    metaestabilidad.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Lista de aristas (triangulo superior) y conjunto para duplicados.
    iu = np.argwhere(np.triu(A, 1) > 0)
    edges = [tuple(e) for e in iu]            # [(u,v), ...] con u<v
    edge_set = set(edges)
    E = len(edges)
    if E < 2:
        return A.copy()

    n_swaps = int(n_swaps_factor * E)
    hechos, intentos, max_intentos = 0, 0, 50 * n_swaps

    while hechos < n_swaps and intentos < max_intentos:
        intentos += 1
        i1 = int(rng.integers(E))
        i2 = int(rng.integers(E))
        if i1 == i2:
            continue
        a, b = edges[i1]
        c, d = edges[i2]
        # Orientacion aleatoria de la segunda arista.
        if rng.random() < 0.5:
            c, d = d, c
        # Evitar nodos compartidos (que generarian bucles o multi-aristas).
        if len({a, b, c, d}) < 4:
            continue
        na = (a, d) if a < d else (d, a)
        nb = (c, b) if c < b else (b, c)
        if na in edge_set or nb in edge_set:
            continue
        # Aplicar el swap.
        edge_set.discard(edges[i1])
        edge_set.discard(edges[i2])
        edge_set.add(na)
        edge_set.add(nb)
        edges[i1] = na
        edges[i2] = nb
        hechos += 1

    R = np.zeros_like(A, dtype=np.float64)
    for (u, v) in edge_set:
        R[u, v] = 1.0
        R[v, u] = 1.0
    return R
