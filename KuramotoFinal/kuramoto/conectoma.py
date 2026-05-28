"""
kuramoto.conectoma
==================

Lectura y preparacion del conectoma estructural humano.

Datos:
    SCmatrices88healthy.mat (Domhof et al. 2022): 88 matrices 90x90, una
    por sujeto sano, con la conectividad estructural entre las 90 regiones
    del atlas AAL.

Funciones:

    cargar_conectoma_promedio(ruta)
        Promedia los 88 sujetos -> matriz de pesos W (90, 90).

    cargar_conectoma_sujetos(ruta)
        Devuelve todas las matrices individuales (88, 90, 90) por si quieres
        analizar la variabilidad entre sujetos en vez del promedio.

    binarizar(W, threshold)
        Aplica corte y devuelve A binaria (0/1), simetrica, diagonal 0.

    es_conexo(A)
        Comprueba si el grafo de A es conexo (BFS con producto matriz-vector).

    buscar_threshold_optimo(W, ...)
        Encuentra el threshold MAS ALTO que aun mantiene el grafo conexo.

    cargar_y_preparar_A(ruta, threshold='auto')
        Pipeline completo: leer .mat -> simetrizar -> threshold optimo
        (o explicito) -> binarizar -> devolver A lista para Kuramoto.

Esta es la unica parte donde la Tarea 4 necesita codigo especifico. El
resto del paquete (system, sweep, plotting) trabaja igual con A modular
sintetica que con A del conectoma.
"""

import numpy as np
import scipy.io as sio


# ----------------------------------------------------------------------------
# Lectura del .mat
# ----------------------------------------------------------------------------

def cargar_conectoma_sujetos(ruta_mat, clave='SCmatrices'):
    """Devuelve las matrices INDIVIDUALES de los 88 sujetos.

    Parametros
    ----------
    ruta_mat : str
        Ruta al archivo .mat.
    clave : str
        Nombre de la variable dentro del .mat. En el dataset estandar de
        Domhof es 'SCmatrices'. Lo dejamos parametrizable por si tienes
        otro .mat con otra clave (Tarea 4 alternativa).

    Devuelve
    --------
    matrices : ndarray, shape (88, 90, 90)
    """
    data     = sio.loadmat(ruta_mat)
    matrices = data[clave]
    return matrices


def cargar_conectoma_promedio(ruta_mat, clave='SCmatrices'):
    """Carga las matrices del .mat, las promedia y devuelve W (90, 90).

    El promedio reduce ruido individual y captura la "estructura tipica"
    del conectoma sano. Es la matriz de pesos sobre la que se aplica el
    threshold para construir A binaria.
    """
    matrices = cargar_conectoma_sujetos(ruta_mat, clave=clave)
    # axis=0 colapsa el eje de los sujetos: (88,90,90) -> (90,90).
    W = np.mean(matrices, axis=0)
    return W


# ----------------------------------------------------------------------------
# Binarizacion con threshold
# ----------------------------------------------------------------------------

def binarizar(W, threshold):
    """Convierte W de pesos en A de adyacencia binaria.

        A[i, j] = 1   si  W_sim[i, j] > threshold
        A[i, j] = 0   en caso contrario

    Pasos:
        1. Simetrizar: W_sim = (W + W.T) / 2. La conectividad estructural
           es fisicamente no direccional; cualquier asimetria en W viene
           del ruido de tractografia. La media de los dos sentidos elimina
           esa asimetria sin alterar la conectividad real.
        2. Aplicar threshold y convertir a uint8 (1 byte) para ahorrar
           memoria frente a int64.
        3. Forzar diagonal a 0 (sin auto-acoplamiento).

    Por que un threshold pequeno:
        En el conectoma estructural muchas entradas debiles son ruido de
        tractografia: fibras detectadas de forma poco fiable. Cortando
        nos quedamos con las conexiones que aparecen de forma consistente.

    Devuelve
    --------
    A : ndarray (N, N), uint8. Simetrica, diagonal nula.
    """
    W_sim = 0.5 * (W + W.T)
    A     = (W_sim > threshold).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A


# ----------------------------------------------------------------------------
# Conectividad del grafo
# ----------------------------------------------------------------------------

def es_conexo(A):
    """True si el grafo de A es conexo, False si tiene varias componentes.

    Por que importa:
        En Kuramoto, si el grafo se desconecta cada componente evoluciona
        independientemente. La sincronizacion no puede propagarse entre
        componentes, por mucho que subamos K. Un conectoma realista debe
        ser conexo: queremos modelar UN cerebro, no varios aislados.

    Algoritmo (BFS implicito via producto matriz-vector):
        Empezamos con el nodo 0 marcado como visitado. En cada paso,
        A.dot(visitados) cuenta los vecinos visitados de cada nodo:
        si i tiene al menos uno, debe pasar a "visitado". Iteramos hasta
        que no cambie nada. Si al final TODOS estan visitados, el grafo
        es conexo (todo es alcanzable desde el nodo 0).
    """
    N = A.shape[0]
    visitados = np.zeros(N, dtype=bool)
    visitados[0] = True

    while True:
        nuevos = visitados | (A.dot(visitados.astype(np.uint8)) > 0)
        if np.array_equal(nuevos, visitados):
            break
        visitados = nuevos

    return bool(visitados.all())


# ----------------------------------------------------------------------------
# Busqueda del threshold optimo
# ----------------------------------------------------------------------------

def buscar_threshold_optimo(W, n_thresholds=300, verbose=True):
    """Encuentra el threshold MAS ALTO que aun deja el grafo conexo.

    Criterio:
        - Threshold demasiado BAJO -> red casi completa (sin estructura util).
        - Threshold demasiado ALTO -> grafo fragmentado (sin sentido fisico).
        - El "punto justo" -> el mas alto que aun da un grafo de una sola
          componente. Asi limpiamos el maximo ruido de tractografia sin
          perder la conectividad global.

    Devuelve
    --------
    thr_optimo : float
    info : dict
        'thresholds'  : array de thresholds probados (1e-6 .. W.max())
        'densidades'  : densidad del grafo en cada threshold
        'conexos'     : bool array, True si el grafo es conexo
    """
    # Empezamos en epsilon > 0: thr = 0 incluye todos los pesos no nulos
    # y siempre da conexo. No es informativo.
    thresholds = np.linspace(1e-6, W.max(), n_thresholds)

    densidades = []
    conexos    = []
    N        = W.shape[0]
    n_pares  = N * (N - 1) // 2

    for thr in thresholds:
        A = binarizar(W, thr)
        # A es simetrica con diagonal 0; A.sum() cuenta cada arista 2 veces.
        n_enlaces = int(A.sum() // 2)
        densidades.append(n_enlaces / n_pares)
        conexos.append(es_conexo(A))

    densidades = np.array(densidades)
    conexos    = np.array(conexos)

    indices_conexos = np.where(conexos)[0]
    if len(indices_conexos) == 0:
        raise ValueError("Ningun threshold del barrido da grafo conexo. "
                         "Revisar W: puede que este vacio o muy ralo.")

    indice_optimo   = indices_conexos.max()
    thr_optimo      = thresholds[indice_optimo]
    densidad_optima = densidades[indice_optimo]

    if verbose:
        print(f"Threshold optimo: {thr_optimo:.6f}")
        print(f"  Densidad: {densidad_optima:.4f}")
        print(f"  N enlaces: {int(densidad_optima * n_pares)}")

    info = {
        'thresholds': thresholds,
        'densidades': densidades,
        'conexos'   : conexos,
    }
    return thr_optimo, info


# ----------------------------------------------------------------------------
# Pipeline completo: .mat -> A lista para usar
# ----------------------------------------------------------------------------

def cargar_y_preparar_A(ruta_mat, threshold='auto', clave='SCmatrices',
                         devolver_W=False, verbose=True):
    """Atajo: lee .mat, calcula W promedio, busca threshold y binariza.

    Parametros
    ----------
    ruta_mat : str
        Ruta al archivo .mat.
    threshold : float | 'auto'
        Si 'auto', busca el threshold optimo (mas alto que mantiene conexo).
        Si float, se usa tal cual.
    clave : str
        Nombre de la variable dentro del .mat.
    devolver_W : bool
        Si True, devuelve tambien la matriz W de pesos sin binarizar.
    verbose : bool
        Imprime resumen al construir.

    Devuelve
    --------
    A : ndarray (N, N), float64 (lista para pasar a Kuramoto).
    (Opcionalmente W).
    thr : float (threshold finalmente usado).
    """
    W = cargar_conectoma_promedio(ruta_mat, clave=clave)

    if isinstance(threshold, str) and threshold == 'auto':
        thr, _ = buscar_threshold_optimo(W, verbose=verbose)
    else:
        thr = float(threshold)
        if verbose:
            print(f"Usando threshold explicito: {thr:.6f}")

    A = binarizar(W, thr).astype(np.float64)  # float64 para encajar con el integrador

    if verbose:
        N         = A.shape[0]
        n_pares   = N * (N - 1) // 2
        n_enlaces = int(A.sum() // 2)
        print(f"Conectoma preparado: N = {N}, |E| = {n_enlaces}, "
              f"densidad = {n_enlaces / n_pares:.4f}")

    if devolver_W:
        return A, W, thr
    return A, thr
