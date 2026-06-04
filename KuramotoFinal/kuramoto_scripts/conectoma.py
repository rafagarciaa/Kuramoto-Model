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
import re
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


def cargar_y_preparar_A(ruta_mat, threshold , clave='SCmatrices', verbose=True):
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
# Lobulos (clasificacion de regiones AAL por lobulo cerebral, sim_type 5)
# ----------------------------------------------------------------------------

# Orden y nombre de los lobulos. Los indices en el array que devuelve
# `lobulo_ids` se corresponden con las posiciones de esta lista.
LOBE_NAMES = [
    'frontal',         # 0  cortex frontal (incluye motor primario)
    'insular_limbico', # 1  insula, cingulum, hipocampo, amigdala
    'occipital',       # 2  cortex visual + fusiform
    'parietal',        # 3  cortex parietal + somatosensorial + precuneus
    'subcortical',     # 4  ganglios basales y talamo
    'temporal',        # 5  cortex temporal y Heschl (auditivo)
]

# Frecuencias intrinsecas tipicas asociadas a los lobulos (en "unidades de
# modelo", no Hz). Capturan QUALITATIVAMENTE las relaciones del cerebro
# real: frontal/sensoriomotor beat mas rapido (beta-like), occipital/parietal
# en torno a alpha, temporal/limbico mas lento (theta-like). Valores
# pensados para sigma_intra ~ 0.1: cada lobulo es una distribucion
# estrecha en torno a su mean.
LOBE_DEFAULT_OMEGAS = {
    'frontal':         1.50,
    'insular_limbico': 0.70,
    'occipital':       1.00,
    'parietal':        1.00,
    'subcortical':     0.90,
    'temporal':        0.80,
}

# Reglas de clasificacion: (lista de keywords en mayuscula) -> indice de lobulo.
# Se aplica la PRIMERA regla que matchee, asi que el orden importa.
_LOBE_RULES = [
    # subcortical primero (es el mas especifico)
    (['CAUDATE', 'PUTAMEN', 'PALLIDUM', 'THALAMUS'],                4),
    # insular/limbico
    (['INSULA', 'CINGULUM', 'HIPPOCAMPUS', 'PARAHIPPOCAMPAL',
      'AMYGDALA'],                                                  1),
    # occipital
    (['CALCARINE', 'CUNEUS', 'LINGUAL', 'OCCIPITAL', 'FUSIFORM'],  2),
    # parietal (incluye somatosensorial postcentral)
    (['POSTCENTRAL', 'PARIETAL', 'SUPRAMARGINAL', 'ANGULAR',
      'PRECUNEUS', 'PARACENTRAL'],                                  3),
    # temporal (incluye Heschl/Heschls = cortex auditivo)
    (['HESCHL', 'HESCHLS', 'TEMPORAL'],                             5),
    # frontal (catch-all para el resto: precentral, frontal, orbital,
    # IFG, rolandic operculum, olfactory, rectal, medial gyrus)
    (['PRECENTRAL', 'FRONTAL', 'ORBITAL', 'IFG', 'ROLANDIC',
      'OLFACTORY', 'RECTAL', 'MEDIAL GYRUS'],                       0),
]


def lobulo_ids(ruta_csv):
    """Clasifica cada region AAL en uno de 6 lobulos cerebrales.

    Lobulos (indices 0..5):
        0 = frontal           (precentral, frontal, IFG, orbital, ...)
        1 = insular_limbico   (insula, cingulum, hippocampus, ...)
        2 = occipital         (calcarine, cuneus, lingual, fusiform, ...)
        3 = parietal          (postcentral, parietal, precuneus, ...)
        4 = subcortical       (caudate, putamen, pallidum, thalamus)
        5 = temporal          (Heschl, temporal)

    Cualquier region que no encaje en ninguna regla acaba en 'frontal'
    como catch-all defensivo.

    Devuelve
    --------
    lob_ids : (N,) int64
    """
    ids = []
    with open(ruta_csv, 'r', encoding='utf-8') as f:
        lector = csv.reader(f, delimiter=';')
        next(lector, None)
        for fila in lector:
            if len(fila) < 2:
                continue
            nombre = fila[1].strip().upper()
            # Quitamos el prefijo 'L ' o 'R ' (hemisferio).
            if nombre.startswith('L ') or nombre.startswith('R '):
                nombre = nombre[2:].strip()
            asignado = 0  # default: frontal
            for keywords, lob_idx in _LOBE_RULES:
                # Matching de palabra COMPLETA (\b regex) para evitar
                # falsos positivos tipo "ANGULAR" en "TRIANGULARIS".
                if any(re.search(r'\b' + re.escape(kw) + r'\b', nombre)
                       for kw in keywords):
                    asignado = lob_idx
                    break
            ids.append(asignado)
    return np.asarray(ids, dtype=np.int64)


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


# ============================================================================
# Tipo 4: conectoma con pesos (sin binarizar)
# ============================================================================
#
# La ecuacion dinamica es la misma de la red:
#
#     theta_dot_i = omega_i + K * sum_j W_ij * sin(theta_j - theta_i)
#
# pero W ya NO es binaria. K es entonces solo un multiplicador escalar y
# son los pesos los que codifican la fuerza relativa de cada conexion.
#
# Dos preparaciones disponibles:
#
#   prepare_W_real(ruta)            -> W promedio con diagonal a 0, sin
#                                      tocar valores.
#   prepare_W_intervalos(ruta, ...) -> W normalizada y aproximada a una
#                                      rejilla de n_levels valores en [0,1].
#                                      Optionally log-transform antes de
#                                      normalizar para aplanar la enorme
#                                      asimetria de la distribucion de pesos
#                                      del conectoma.
# ----------------------------------------------------------------------------

def prepare_W_real(ruta_mat, clave='SCmatrices'):
    """W = promedio de los 88 sujetos, simetrizada y con diagonal a 0.

    Simetrizamos con (W + W.T)/2 porque el conectoma es fisicamente no
    direccional: cualquier asimetria proviene del ruido de tractografia.
    La diagonal valia 1.0 en el dataset original (autoconexion trivial),
    la forzamos a 0 para Kuramoto.
    """
    W = cargar_conectoma_promedio(ruta_mat, clave=clave).astype(np.float64)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def prepare_W_intervalos(ruta_mat, n_levels, log_transform=True,
                          clave='SCmatrices'):
    """W promedio con diagonal=0, normalizada a [0,1] y aproximada a una
    rejilla de `n_levels` valores equispaciados.

    Para `n_levels = N` los valores posibles son {0, 1/(N-1), 2/(N-1), ..., 1}.
    Ejemplos: N=2 -> {0, 1}; N=3 -> {0, 0.5, 1}; N=5 -> {0, 0.25, 0.5, 0.75, 1}.

    Cada entrada de W_norm se redondea al valor mas cercano de la rejilla.

    log_transform:
        El conectoma tiene una distribucion extremadamente sesgada (mediana
        de los no-nulos ~1e-4, max=1.0). Si normalizas directamente por max,
        casi todo cae cerca de 0 y los niveles altos quedan vacios. Aplicar
        log(1+W) antes de normalizar comprime el rango dinamico y reparte la
        masa entre los niveles. Recomendado True para el conectoma; False
        si W ya tiene distribucion bien escalada.
    """
    W = cargar_conectoma_promedio(ruta_mat, clave=clave).astype(np.float64)
    W = 0.5 * (W + W.T)              # simetrizar primero (ruido tractografia)
    np.fill_diagonal(W, 0.0)

    if log_transform:
        W = np.log1p(W)              # log(1+W): comprime range dinamico

    W_max = float(W.max())
    if W_max <= 0:
        return W.astype(np.float64)  # matriz vacia: nada que aproximar
    W_norm = W / W_max                # ahora valores en [0, 1]

    # Aproximacion a la rejilla {0, 1/(n-1), ..., 1}.
    step = 1.0 / (n_levels - 1)
    W_approx = np.round(W_norm * (n_levels - 1)) * step
    np.fill_diagonal(W_approx, 0.0)   # garantizamos que la diagonal sigue 0
    return W_approx.astype(np.float64)


def randomize_preserving_strength(W, n_swaps_factor=20, rng=None):
    """Devuelve una red aleatoria con misma strength por nodo que W.

    Strength del nodo i = s_i = sum_j W_ij.

    Algoritmo: 4-cycle swap (swap-balanced) que preserva strength EXACTO.
    Repite la siguiente jugada n_swaps = n_swaps_factor * |E_nonzero|:
        - Elegir 4 nodos distintos i, j, k, l al azar.
        - Elegir un epsilon en [0, min(W_ij, W_kl)].
        - W_ij -= eps, W_kl -= eps, W_il += eps, W_kj += eps.
          (Simetrico, es decir tambien W_ji, W_lk, W_li, W_jk.)
    Esto conserva s_i, s_j, s_k, s_l (cada nodo gana y pierde lo mismo).

    Compara con `randomize_preserving_degree`: aquella conserva la
    estructura binaria (cada nodo tiene mismo numero de vecinos);
    esta conserva strength en grafos con pesos arbitrarios.
    """
    if rng is None:
        rng = np.random.default_rng()

    R = W.copy().astype(np.float64)
    np.fill_diagonal(R, 0.0)
    N = R.shape[0]
    E_nonzero = int((R > 0).sum() // 2)     # aristas para escalar n_swaps
    if E_nonzero < 2 or N < 4:
        return R

    n_swaps = int(n_swaps_factor * E_nonzero)
    hechos, intentos = 0, 0
    max_intentos = 50 * n_swaps

    while hechos < n_swaps and intentos < max_intentos:
        intentos += 1
        # Elegir 4 nodos distintos.
        idx = rng.choice(N, size=4, replace=False)
        i, j, k, l = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])

        wij, wkl = R[i, j], R[k, l]
        # Solo tiene sentido si al menos una de las dos aristas tiene peso.
        # Si ambas son 0, no podemos restar nada.
        max_eps = min(wij, wkl)
        if max_eps <= 0:
            continue

        eps = float(rng.uniform(0.0, max_eps))
        # Aplicar swap simetrico.
        R[i, j] -= eps;  R[j, i] -= eps
        R[k, l] -= eps;  R[l, k] -= eps
        R[i, l] += eps;  R[l, i] += eps
        R[k, j] += eps;  R[j, k] += eps
        hechos += 1

    return R
