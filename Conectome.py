"""
conectoma.py
============

Dos funciones basicas para trabajar con el conectoma estructural humano:

    cargar_conectoma_promedio(ruta)  -> devuelve la matriz promedio (90, 90)
    visualizar_matriz(W)              -> dibuja la matriz como imagen

Los datos vienen de SCmatrices88healthy.mat (Domhof et al. 2022), que
contiene 88 matrices 90x90 (una por sujeto sano) con la conectividad
estructural entre las 90 regiones del atlas AAL.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# ===========================================================================
# Funcion 1: cargar el .mat y devolver la matriz promedio
# ===========================================================================
def cargar_conectoma_promedio(ruta_mat):
    """
    Carga las 88 matrices del .mat, las promedia y devuelve la matriz
    promedio de tamaño (90, 90).

    Parametros
    ----------
    ruta_mat : str
        Ruta al archivo SCmatrices88healthy.mat.

    Devuelve
    --------
    W : ndarray, shape (90, 90), dtype float64
        Matriz de conectividad promedio de los 88 sujetos.
    """
    # ----- Paso 1: leer el .mat ---------------------------------------------
    # scipy.io.loadmat lee archivos en formato MATLAB y devuelve un
    # diccionario. Las claves son los nombres de las variables guardadas
    # en el archivo original. En este caso la unica variable util se
    # llama 'SCmatrices' (lo sabemos porque lo dice el paper / readme).
    data = sio.loadmat(ruta_mat)

    # Extraemos el array de matrices. Forma esperada: (88, 90, 90)
    #   - eje 0: indice del sujeto (0..87)
    #   - ejes 1 y 2: filas y columnas de la matriz 90x90 de ese sujeto
    matrices = data['SCmatrices']

    # ----- Paso 2: promediar a lo largo del eje de los sujetos --------------
    # np.mean(array, axis=0) calcula la media a lo largo del eje indicado,
    # colapsandolo. Aqui axis=0 es el eje de los sujetos, asi que pasamos
    # de (88, 90, 90) a (90, 90): cada entrada (i, j) de la matriz
    # resultante es la media de los 88 valores W_k[i, j] de los sujetos.
    W = np.mean(matrices, axis=0)

    return W


# ===========================================================================
# Funcion 2: aplicar threshold y devolver matriz de adyacencia binaria
# ===========================================================================
def binarizar(W, threshold):
    """
    Convierte la matriz de pesos W en una matriz de adyacencia binaria A:
        A[i, j] = 1   si  W[i, j] > threshold
        A[i, j] = 0   en caso contrario

    Ademas fuerza la diagonal de A a 0 (sin auto-conexiones), porque en
    el modelo de Kuramoto un nodo no se acopla consigo mismo. Esto es
    independiente de lo que vaya en la diagonal de W: la matriz de
    adyacencia "de trabajo" siempre tiene diagonal nula por definicion.

    Por que un threshold pequeño:
        En el conectoma estructural hay muchas entradas muy debiles que
        suelen ser "ruido de tractografia": fibras detectadas de forma
        poco fiable que no representan conexiones reales. Cortando con
        un threshold pequeño descartamos ese ruido y nos quedamos con
        las conexiones que aparecen de forma consistente en el promedio
        sobre los 88 sujetos.

    Parametros
    ----------
    W : ndarray, shape (N, N)
        Matriz de pesos (la que devuelve cargar_conectoma_promedio).
    threshold : float
        Valor de corte. Entradas estrictamente mayores -> enlace (1),
        el resto -> no enlace (0).

    Devuelve
    --------
    A : ndarray, shape (N, N), dtype uint8
        Matriz de adyacencia binaria, con diagonal nula.
    """
    # ----- Paso 1: simetrizar W ---------------------------------------------
    # El conectoma estructural es fisicamente NO direccional: las fibras
    # blancas no tienen sentido preferido. Sin embargo, la matriz que
    # devuelve la tractografia puede no ser exactamente simetrica por
    # ruido numerico (W[i,j] ligeramente distinto de W[j,i]).
    # Promediamos las dos direcciones para eliminar esa asimetria:
    #     W_sim[i,j] = (W[i,j] + W[j,i]) / 2
    # Si W ya era simetrica, esto no cambia nada. Si no lo era, queda
    # simetrica y representa la "evidencia media" de conexion.
    W_sim = 0.5 * (W + W.T)

    # ----- Paso 2: aplicar threshold ----------------------------------------
    # Como funciona:
    #   (W_sim > threshold)  -> array booleano (True/False) elemento a elemento.
    #   .astype(np.uint8) -> convierte True->1 y False->0.
    #
    # Usamos uint8 (entero de 1 byte) en vez de int64 (8 bytes) porque
    # solo necesitamos guardar 0s y 1s. Asi ocupa 8 veces menos memoria
    # y, mas importante, los productos numpy con la matriz son mas rapidos.
    A = (W_sim > threshold).astype(np.uint8)

    # ----- Paso 3: diagonal a 0 ---------------------------------------------
    # Sin auto-conexiones. fill_diagonal modifica A "in-place".
    np.fill_diagonal(A, 0)

    return A


# ===========================================================================
# Funcion auxiliar: comprobar si un grafo es conexo
# ===========================================================================
def es_conexo(A):
    """
    Devuelve True si la matriz de adyacencia A representa un grafo conexo,
    False si esta dividido en varias componentes desconectadas.

    Por que esto importa:
        En Kuramoto, si el grafo se desconecta cada componente evoluciona
        independientemente. No hay forma de que la sincronizacion se
        propague entre componentes, por mucho que aumentemos K. Por eso
        un conectoma realista debe ser conexo: queremos un solo cerebro,
        no varios cerebros aislados.

    Como funciona el algoritmo (BFS implicito):
        - Empezamos con el nodo 0 "visitado" y los demas "no visitados".
        - En cada iteracion, marcamos como visitados todos los vecinos
          de los nodos ya visitados.
        - Repetimos hasta que no cambie nada.
        - El grafo es conexo si y solo si al final TODOS los nodos
          quedaron visitados (es decir, alcanzables desde el nodo 0).

        La propagacion la hacemos con un producto matriz-vector:
            A @ visitados
        da, en la entrada i, el numero de vecinos visitados de i.
        Si i tiene al menos un vecino visitado, esa entrada es > 0.

    Parametros
    ----------
    A : ndarray, shape (N, N)
        Matriz de adyacencia (simetrica, diagonal nula, 0/1).

    Devuelve
    --------
    bool : True si el grafo es conexo, False si no.
    """
    N = A.shape[0]

    # Vector booleano: visitados[i] = True si el nodo i es alcanzable
    # desde el nodo 0. Inicialmente, solo el nodo 0 esta visitado.
    visitados = np.zeros(N, dtype=bool)
    visitados[0] = True

    # Iteramos hasta que no se descubran nodos nuevos.
    while True:
        # A.dot(visitados.astype(uint8)) da, en cada entrada i, cuantos
        # vecinos visitados tiene i. Si > 0, i debe pasar a "visitado".
        nuevos = visitados | (A.dot(visitados.astype(np.uint8)) > 0)

        # Si no hemos descubierto ningun nodo nuevo, paramos.
        if np.array_equal(nuevos, visitados):
            break
        visitados = nuevos

    # El grafo es conexo si TODOS los nodos quedaron visitados.
    return bool(visitados.all())


# ===========================================================================
# Funcion: buscar el threshold optimo
# ===========================================================================
def buscar_threshold_optimo(W, n_thresholds=300, mostrar_plot=True):
    """
    Encuentra el threshold mas alto posible que aun mantiene el grafo
    de adyacencia conexo.

    Justificacion del criterio:
        - Un threshold demasiado bajo deja casi todos los pares como
          enlaces -> red casi completa, sin estructura interesante.
        - Un threshold demasiado alto fragmenta el grafo -> deja de
          tener sentido como modelo de cerebro completo.
        - El "punto justo" es el threshold mas alto donde el grafo
          aun forma una sola componente. Asi limpiamos el maximo ruido
          de tractografia sin perder la propiedad fisica fundamental.

    Como lo hace:
        1. Barre `n_thresholds` valores entre 0 y W.max().
        2. Para cada valor calcula:
             - la matriz de adyacencia binarizada.
             - su densidad (enlaces / pares posibles).
             - si el grafo resultante es conexo.
        3. Devuelve el threshold mas alto que da grafo conexo.

    Parametros
    ----------
    W : ndarray, shape (N, N)
        Matriz de pesos.
    n_thresholds : int
        Numero de puntos del barrido. 300 es mas que suficiente para 90 nodos.
    mostrar_plot : bool
        Si True, dibuja la curva densidad vs threshold y marca el optimo.

    Devuelve
    --------
    thr_optimo : float
        Threshold optimo.
    info : dict
        Diccionario con los arrays del barrido por si quieres inspeccionarlos:
            'thresholds'  : array de thresholds probados
            'densidades'  : densidad del grafo para cada threshold
            'conexos'     : bool array (True/False) por cada threshold
    """
    # Barrido de thresholds entre un valor muy pequeño y el maximo de W.
    # Empezamos en un epsilon pequeño (no en 0) porque thr = 0 incluye
    # todos los pesos no nulos como enlace, que ya sabemos que da grafo conexo.
    thresholds = np.linspace(1e-6, W.max(), n_thresholds)

    # Vamos guardando los resultados en listas (mas simple que arrays
    # de tamaño fijo si quisieramos parar antes).
    densidades = []
    conexos = []

    N = W.shape[0]
    n_pares = N * (N - 1) // 2   # numero de pares posibles (i < j)

    # Bucle principal del barrido
    for thr in thresholds:
        A = binarizar(W, thr)

        # A es simetrica con diagonal 0. A.sum() cuenta los 1s en TODA
        # la matriz, asi que cada enlace se cuenta 2 veces.
        n_enlaces = int(A.sum() // 2)
        densidades.append(n_enlaces / n_pares)

        conexos.append(es_conexo(A))

    # Convertimos a arrays para poder hacer slicing / np.where.
    densidades = np.array(densidades)
    conexos = np.array(conexos)

    # ----- Encontrar el threshold optimo ------------------------------------
    # np.where(conexos)[0] da los indices donde conexos == True.
    # Cogemos el MAYOR indice -> el threshold mas alto que aun es conexo.
    indices_conexos = np.where(conexos)[0]
    if len(indices_conexos) == 0:
        raise ValueError("Ningun threshold del barrido da grafo conexo. "
                         "Algo raro pasa con W.")
    indice_optimo = indices_conexos.max()
    thr_optimo = thresholds[indice_optimo]
    densidad_optima = densidades[indice_optimo]

    # ----- Plot opcional ----------------------------------------------------
    if mostrar_plot:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Curva densidad vs threshold. La pintamos en dos colores:
        # azul donde el grafo es conexo, rojo donde no lo es.
        ax.plot(thresholds[conexos], densidades[conexos], 'b.',
                markersize=4, label='Conexo')
        ax.plot(thresholds[~conexos], densidades[~conexos], 'r.',
                markersize=4, label='Desconexo')

        # Marca vertical en el threshold optimo.
        ax.axvline(thr_optimo, color='k', linestyle='--', linewidth=1,
                   label=f'Optimo = {thr_optimo:.5f}')
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Densidad de la red")
        ax.set_title("Busqueda del threshold optimo\n"
                     "(maximo que mantiene el grafo conexo)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Imprimimos un pequeño resumen.
    print(f"Threshold optimo: {thr_optimo:.6f}")
    print(f"  Densidad correspondiente: {densidad_optima:.4f}")
    print(f"  Numero de enlaces: {int(densidad_optima * n_pares)}")

    info = {
        'thresholds': thresholds,
        'densidades': densidades,
        'conexos': conexos,
    }
    return thr_optimo, info


# ===========================================================================
# Funcion: visualizar una matriz como imagen
# ===========================================================================
def visualizar_matriz(W, titulo="Conectoma"):
    """
    Dibuja la matriz W como una imagen en escala de grises.

    Cada celda (i, j) es un pixel cuya intensidad (mas oscura = mas grande)
    representa el valor W[i, j]. Es la forma estandar de visualizar una
    matriz de conectividad.

    Parametros
    ----------
    W : ndarray, shape (N, N)
        Matriz a visualizar.
    titulo : str
        Titulo del grafico.
    """
    # plt.subplots crea una figura y unos "ejes" (los ejes son donde
    # se dibuja). figsize esta en pulgadas (ancho, alto).
    fig, ax = plt.subplots(figsize=(7, 7))

    # ax.imshow muestra la matriz como imagen.
    #   cmap="Greys": colormap. 0 -> blanco, max -> negro.
    #     (asi enlaces fuertes salen oscuros, los debiles claros)
    #   interpolation="nearest": no suaviza entre pixeles vecinos.
    #     Cada celda se ve como un cuadrado bien definido.
    #   aspect="equal": pixeles cuadrados (no se estira la imagen).
    # imshow devuelve un objeto "imagen" que guardamos en `im` para poder
    # asociarle luego una barra de color.
    im = ax.imshow(W, cmap="Greys", interpolation="nearest", aspect="equal")

    # Barra de color a la derecha, para saber a que valor corresponde
    # cada tono de gris.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Etiquetas. Las regiones del atlas AAL estan ordenadas por
    # hemisferios (primero izquierdo, luego derecho), asi que en el plot
    # se deberian apreciar dos bloques mas densos en la diagonal
    # (conectividad intra-hemisferio).
    ax.set_title(titulo)
    ax.set_xlabel("Region (AAL)")
    ax.set_ylabel("Region (AAL)")

    # tight_layout ajusta margenes para que no se solapen los textos.
    plt.tight_layout()

    # plt.show() abre la ventana del grafico. Si estas en un notebook,
    # el plot aparece automaticamente; si estas en script, esta linea
    # es la que abre la ventana.
    plt.show()


# ===========================================================================
# Bloque main: solo se ejecuta si llamas "python conectoma.py" directamente
# ===========================================================================
if __name__ == "__main__":
    # Ruta al archivo .mat. Cambia esto si lo guardas en otro sitio.
    RUTA = "SCmatrices88healthy.mat"

    # ----- Matriz de pesos -------------------------------------------------
    W = cargar_conectoma_promedio(RUTA)
    print("Matriz promedio cargada.")
    print("  Shape:", W.shape)
    print("  Min, max:", W.min(), W.max())
    print()

    # ----- Buscar el threshold optimo --------------------------------------
    # El optimo se define como el mas alto que aun mantiene el grafo conexo.
    thr_optimo, info = buscar_threshold_optimo(W, n_thresholds=300)

    # ----- Matriz de adyacencia con el threshold optimo --------------------
    A = binarizar(W, threshold=thr_optimo)
    visualizar_matriz(A, titulo=f"Matriz de adyacencia optima "
                                f"(threshold = {thr_optimo:.5f})")