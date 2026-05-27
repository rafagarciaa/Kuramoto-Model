"""
Lectura de estructuras modulares jerárquicas y construcción de matrices
de adyacencia a partir de ellas.

Formato de la estructura (ver crear_estructura.py):
    Cada módulo interno es una lista [M, hijo_1, ..., hijo_n], donde M
    es una matriz simétrica n x n cuya diagonal contiene los tamaños de
    los hijos y cuyas entradas (i, j) con i != j indican el número de
    enlaces aleatorios entre los hijos i y j a este nivel.
    Una hoja se representa como null (None en Python).
"""

import json
from pathlib import Path

import numpy as np


def cargar_estructura(ruta):
    """Carga una estructura desde un archivo JSON.

    Parameters
    ----------
    ruta : str o Path
        Ruta al archivo JSON. Si no termina en .json se asume que es un
        nombre dentro de la carpeta estructuras/ junto a este módulo.

    Returns
    -------
    estructura : list o None
        El árbol modular en el formato descrito arriba.
    """
    ruta = Path(ruta)
    if not ruta.suffix:
        ruta = ruta.with_suffix(".json")
    if not ruta.is_absolute() and not ruta.exists():
        ruta = Path(__file__).resolve().parent / "estructuras" / ruta.name
    with open(ruta) as f:
        return json.load(f)


def construir_matriz_adyacencia(estructura, N, rng=None):
    """Genera UNA realización aleatoria de la matriz de adyacencia.

    Cada llamada con un `rng` distinto produce una realización distinta:
    las hojas son fully-connected (deterministas), pero los enlaces
    inter-módulo se colocan al azar entre los nodos de los hermanos
    correspondientes.

    Parameters
    ----------
    estructura : list o None
        Árbol modular cargado con `cargar_estructura` (o construido a mano).
    N : int
        Número total de osciladores. Debe coincidir con la suma de los
        tamaños declarados en la raíz (se verifica).
    rng : np.random.Generator, opcional
        Generador de números aleatorios. Si es None se crea uno nuevo
        sin semilla.

    Returns
    -------
    A : ndarray, shape (N, N), dtype uint8
        Matriz de adyacencia simétrica, con ceros en la diagonal.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Verificación de tamaños en la raíz
    if estructura is not None:
        M_raiz = estructura[0]
        tamaño_raiz = sum(M_raiz[k][k] for k in range(len(M_raiz)))
        if tamaño_raiz != N:
            raise ValueError(
                f"La estructura suma {tamaño_raiz} osciladores pero N={N}."
            )

    A = np.zeros((N, N), dtype=np.uint8)
    _rellenar(A, estructura, offset=0, tamaño=N, rng=rng)
    return A


def _rellenar(A, nodo, offset, tamaño, rng):
    if nodo is None:
        # Hoja: bloque fully-connected
        bloque = np.ones((tamaño, tamaño), dtype=np.uint8)
        np.fill_diagonal(bloque, 0)
        A[offset:offset + tamaño, offset:offset + tamaño] = bloque
        return

    M = nodo[0]
    hijos = nodo[1:]
    n = len(hijos)
    tamaños_hijos = [M[k][k] for k in range(n)]

    # Rellenar bloques hijos recursivamente
    offsets = [offset]
    for k in range(n):
        offsets.append(offsets[-1] + tamaños_hijos[k])
        _rellenar(A, hijos[k], offset=offsets[k], tamaño=tamaños_hijos[k], rng=rng)

    # Enlaces aleatorios inter-módulo según la triangular superior de M
    for i in range(n):
        for j in range(i + 1, n):
            n_enlaces = M[i][j]
            if n_enlaces == 0:
                continue
            _añadir_enlaces_aleatorios(
                A,
                rango_i=(offsets[i], offsets[i + 1]),
                rango_j=(offsets[j], offsets[j + 1]),
                n_enlaces=n_enlaces,
                rng=rng,
            )


def _añadir_enlaces_aleatorios(A, rango_i, rango_j, n_enlaces, rng):
    """Coloca n_enlaces enlaces sin repetir entre los nodos de
    [rango_i[0], rango_i[1]) y [rango_j[0], rango_j[1])."""
    i0, i1 = rango_i
    j0, j1 = rango_j
    n_i = i1 - i0
    n_j = j1 - j0
    max_posibles = n_i * n_j
    if n_enlaces > max_posibles:
        raise ValueError(
            f"Se piden {n_enlaces} enlaces entre módulos de tamaño {n_i} y "
            f"{n_j}, pero solo hay {max_posibles} pares posibles."
        )

    # Muestreo sin reemplazo de pares (u, v) en el rectángulo i × j
    pares_planos = rng.choice(max_posibles, size=n_enlaces, replace=False)
    us = i0 + pares_planos // n_j
    vs = j0 + pares_planos %  n_j
    A[us, vs] = 1
    A[vs, us] = 1