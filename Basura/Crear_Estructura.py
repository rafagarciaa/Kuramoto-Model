"""
Constructor interactivo de estructuras modulares jerárquicas para
simulaciones de Kuramoto.

Lee N desde parametros.json (en el mismo directorio que este script) y
guía al usuario por las preguntas necesarias para construir un árbol
modular, visualizándolo antes de guardar el JSON resultante en
estructuras/.

Formato del archivo guardado:
    Cada módulo interno es una lista [M, hijo_1, hijo_2, ..., hijo_n],
    donde M es la matriz simétrica de n x n cuya DIAGONAL contiene los
    tamaños de los hijos y cuyas entradas (i, j) con i != j contienen
    el número de enlaces aleatorios entre los hijos i y j.
    Una hoja se representa como null.
"""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
SCRIPT_DIR       = Path(__file__).resolve().parent
PARAMS_FILE      = SCRIPT_DIR / "parametros.json"
ESTRUCTURAS_DIR  = SCRIPT_DIR / "estructuras"


# ---------------------------------------------------------------------------
# Helpers de entrada por consola
# ---------------------------------------------------------------------------
def preguntar_int(prompt, minimo=None, maximo=None):
    while True:
        try:
            v = int(input(prompt).strip())
            if minimo is not None and v < minimo:
                print(f"  Debe ser >= {minimo}.")
                continue
            if maximo is not None and v > maximo:
                print(f"  Debe ser <= {maximo}.")
                continue
            return v
        except ValueError:
            print("  Entrada inválida, escribe un entero.")


def preguntar_float(prompt, minimo=None, maximo=None):
    while True:
        try:
            v = float(input(prompt).strip())
            if minimo is not None and v < minimo:
                print(f"  Debe ser >= {minimo}.")
                continue
            if maximo is not None and v > maximo:
                print(f"  Debe ser <= {maximo}.")
                continue
            return v
        except ValueError:
            print("  Entrada inválida, escribe un número.")


def preguntar_si_no(prompt):
    while True:
        v = input(prompt).strip().upper()
        if v in ("S", "SI", "SÍ"):
            return True
        if v in ("N", "NO"):
            return False
        print("  Responde S o N.")


def preguntar_si_no_enter(prompt):
    """Como preguntar_si_no, pero Enter cuenta como Sí."""
    while True:
        v = input(prompt).strip().upper()
        if v in ("", "S", "SI", "SÍ"):
            return True
        if v in ("N", "NO"):
            return False
        print("  Responde S, N, o Enter para aceptar.")


# ---------------------------------------------------------------------------
# Carga de parámetros globales
# ---------------------------------------------------------------------------
def cargar_parametros():
    if not PARAMS_FILE.exists():
        print(f"ERROR: no se encuentra {PARAMS_FILE}")
        print("Crea un archivo parametros.json en este directorio con al menos:")
        print('  {"N": 1000}')
        raise SystemExit(1)
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    if "N" not in params:
        raise SystemExit("ERROR: parametros.json debe contener la clave 'N'.")
    return params


# ---------------------------------------------------------------------------
# Construcción de la matriz de conexiones inter-módulo de un nivel
# ---------------------------------------------------------------------------
def construir_matriz_conexiones(n_hijos, etiqueta, enlaces_globales=None):
    """Devuelve una matriz n x n (lista de listas) con las conexiones
    inter-módulo a este nivel. La diagonal queda a 0; el llamador la
    rellenará con los tamaños.

    Si enlaces_globales no es None, se aplica directamente ese valor a
    todos los pares de submódulos sin preguntar al usuario."""
    M = [[0] * n_hijos for _ in range(n_hijos)]
    if n_hijos < 2:
        return M

    # Atajo: si el usuario fijó un alpha global al inicio, lo usamos sin preguntar.
    if enlaces_globales is not None:
        for i in range(n_hijos):
            for j in range(i + 1, n_hijos):
                M[i][j] = M[j][i] = enlaces_globales
        return M

    print(f"\n  Conexiones entre submódulos de [{etiqueta}]:")
    todos = preguntar_si_no("    ¿Conectar todos los submódulos por defecto? [S/N]: ")

    if todos:
        alpha = preguntar_int("    Número de enlaces entre cada par: ", minimo=0)
        for i in range(n_hijos):
            for j in range(i + 1, n_hijos):
                M[i][j] = M[j][i] = alpha
        return M

    # No todos → elegir modo
    print("    Modos disponibles:")
    print("      1) Aleatorio  — cada par se conecta con cierta probabilidad")
    print("      2) Custom     — defines cada par a mano")
    while True:
        opcion = input("    Elige modo [1/2]: ").strip()
        if opcion in ("1", "2"):
            break
        print("    Responde 1 o 2.")

    if opcion == "1":
        p = preguntar_float(
            "    Probabilidad de conexión entre cada par [0-1]: ",
            minimo=0.0, maximo=1.0,
        )
        alpha = preguntar_int("    Número de enlaces entre cada par conectado: ", minimo=0)
        conectados = []
        for i in range(n_hijos):
            for j in range(i + 1, n_hijos):
                if random.random() < p:
                    M[i][j] = M[j][i] = alpha
                    conectados.append((i, j))
        if conectados:
            print(f"    Pares conectados al azar: {conectados}")
        else:
            print("    (El azar no conectó ningún par.)")
    else:
        pares = [(i, j) for i in range(n_hijos) for j in range(i + 1, n_hijos)]
        print("    Orden de los pares a introducir:")
        for k, (i, j) in enumerate(pares, start=1):
            print(f"      {k}. ({i}, {j})")
        for (i, j) in pares:
            v = preguntar_int(f"      Enlaces ({i},{j}): ", minimo=0)
            M[i][j] = M[j][i] = v

    return M


# ---------------------------------------------------------------------------
# Construcción recursiva del árbol
# ---------------------------------------------------------------------------
def construir_nodo(N_disponible, etiqueta, enlaces_globales=None):
    """Pregunta cómo construir un módulo con N_disponible osciladores.
    Devuelve None si es hoja, o [M, hijo_1, ..., hijo_n] si es interno."""
    nombre_visible = etiqueta or "raíz"
    print(f"\n──── Módulo [{nombre_visible}] con {N_disponible} osciladores ────")

    if N_disponible < 2 or not preguntar_si_no("  ¿Se subdivide este módulo? [S/N]: "):
        return None

    n_hijos = preguntar_int(
        "  Número de submódulos (>= 2): ",
        minimo=2, maximo=N_disponible,
    )

    # Tamaños: todos menos el último; el último se calcula automáticamente
    print(f"  Reparto de {N_disponible} osciladores entre {n_hijos} submódulos:")
    tamaños = []
    restante = N_disponible
    for k in range(n_hijos - 1):
        max_k = restante - (n_hijos - 1 - k)  # deja >= 1 para cada submódulo restante
        t = preguntar_int(
            f"    Submódulo {k} (quedan {restante} para {n_hijos - k} submódulos): ",
            minimo=1, maximo=max_k,
        )
        tamaños.append(t)
        restante -= t
    tamaños.append(restante)
    print(f"    Submódulo {n_hijos - 1} (automático): {restante}")

    # Matriz de conexiones (off-diagonal) + diagonal con tamaños
    M = construir_matriz_conexiones(n_hijos, nombre_visible, enlaces_globales)
    for k in range(n_hijos):
        M[k][k] = tamaños[k]

    # Recurrir en cada hijo
    hijos = []
    for k in range(n_hijos):
        sub_etiqueta = f"{etiqueta}.{k}" if etiqueta else f"{k}"
        hijos.append(construir_nodo(tamaños[k], sub_etiqueta, enlaces_globales))

    return [M, *hijos]


# ---------------------------------------------------------------------------
# Visualización ASCII del árbol
# ---------------------------------------------------------------------------
def visualizar(estructura, N):
    print(f"raíz ({N} osc.)")
    if estructura is None:
        print(f"└── [{N} osc.]   (hoja única)")
        return
    M = estructura[0]
    n_hijos = len(estructura) - 1
    for k in range(n_hijos):
        _vis(estructura[k + 1], M[k][k], "", k == n_hijos - 1, k)


def _vis(nodo, tamaño, prefijo, es_ultimo, indice):
    conector  = "└── " if es_ultimo else "├── "
    extension = "    " if es_ultimo else "│   "

    if nodo is None:
        print(f"{prefijo}{conector}[{tamaño} osc.]")
        return

    n_hijos = len(nodo) - 1
    M = nodo[0]
    print(f"{prefijo}{conector}módulo_{indice} ({tamaño} osc., {n_hijos} submódulos)")
    for k in range(n_hijos):
        _vis(nodo[k + 1], M[k][k], prefijo + extension, k == n_hijos - 1, k)


# ---------------------------------------------------------------------------
# Matriz de adyacencia de ejemplo (solo para previsualización)
# ---------------------------------------------------------------------------
def construir_matriz_ejemplo(estructura, N):
    """Genera UNA realización de la matriz de adyacencia a partir de la
    estructura. Hojas = bloques fully-connected. Enlaces inter-módulo =
    pares aleatorios. Pensada solo para previsualizar, no para simular."""
    A = np.zeros((N, N), dtype=np.uint8)
    rng = np.random.default_rng()
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

    # Rellenar cada hijo
    offsets = [offset]
    for k in range(n):
        offsets.append(offsets[-1] + tamaños_hijos[k])
        _rellenar(A, hijos[k], offset=offsets[k], tamaño=tamaños_hijos[k], rng=rng)

    # Enlaces inter-módulo
    for i in range(n):
        for j in range(i + 1, n):
            n_enlaces = M[i][j]
            if n_enlaces == 0:
                continue
            nodos_i = np.arange(offsets[i], offsets[i + 1])
            nodos_j = np.arange(offsets[j], offsets[j + 1])
            colocados = 0
            intentos = 0
            limite = 100 * n_enlaces + 100
            while colocados < n_enlaces and intentos < limite:
                u = rng.choice(nodos_i)
                v = rng.choice(nodos_j)
                if A[u, v] == 0:
                    A[u, v] = A[v, u] = 1
                    colocados += 1
                intentos += 1


def visualizar_matriz_ejemplo(A):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(A, cmap="Greys", interpolation="nearest", aspect="equal")
    ax.set_title("Ejemplo de matriz de adyacencia\n(una realización aleatoria)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    print("\n(Cierra la ventana para continuar.)")
    plt.show()


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def guardar(estructura):
    ESTRUCTURAS_DIR.mkdir(exist_ok=True)
    while True:
        nombre = input("\nNombre del archivo (sin extensión): ").strip()
        if not nombre:
            print("  El nombre no puede estar vacío.")
            continue
        if not nombre.endswith(".json"):
            nombre += ".json"
        ruta = ESTRUCTURAS_DIR / nombre
        if ruta.exists():
            if not preguntar_si_no(f"  {ruta.name} ya existe. ¿Sobrescribir? [S/N]: "):
                continue
        break
    with open(ruta, "w") as f:
        json.dump(estructura, f, indent=2)
    print(f"\nEstructura guardada en {ruta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    params = cargar_parametros()
    N = params["N"]

    print("=" * 60)
    print("  Constructor de estructura modular jerárquica")
    print("=" * 60)
    print(f"\nN total (leído de parametros.json): {N}")

    # Pregunta global: ¿el mismo número de enlaces inter-módulo en TODOS los niveles?
    print()
    mismo = preguntar_si_no(
        "¿Todos los módulos en todos los niveles se conectan con el mismo\n"
        "número de enlaces? [S/N]: "
    )
    enlaces_globales = None
    if mismo:
        enlaces_globales = preguntar_int(
            "  Número de enlaces entre cada par de submódulos: ", minimo=0
        )

    while True:
        estructura = construir_nodo(N, "", enlaces_globales)

        print("\n" + "=" * 60)
        print("  Estructura construida")
        print("=" * 60)
        visualizar(estructura, N)
        print()

        # Previsualización gráfica de una matriz de adyacencia de ejemplo
        print("Generando ejemplo de matriz de adyacencia...")
        A_ejemplo = construir_matriz_ejemplo(estructura, N)
        visualizar_matriz_ejemplo(A_ejemplo)

        if preguntar_si_no_enter("¿Conforme con el resultado? [S/N, Enter = S]: "):
            break
        print("\nVolvemos a empezar la construcción.\n")

    guardar(estructura)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelado.")
        raise SystemExit(130)
