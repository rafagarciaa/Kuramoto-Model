"""
kuramoto.io
===========

Utilidades de entrada/salida que NO dependen del modelo en si:

    - Crear la carpeta de resultados sin sobreescribir corridas previas.
    - Volcar el diccionario de parametros a un params.txt legible.
    - Redirigir stdout/stderr a un log.txt sin perder la consola (Tee).

Esta separacion permite que el resto del paquete (system.py, sweep.py,
plotting.py, ...) no se entere de donde se guardan las cosas. Lo unico
que necesitan es una ruta `run_dir` que les pasa el orquestador.
"""

import os
import sys
import time


# Carpeta raiz donde se acumulan todas las ejecuciones. Cada llamada a
# `crear_carpeta_resultados` crea una subcarpeta con timestamp / nombre
# descriptivo dentro de RESULTADOS_BASE. La ruta es RELATIVA al cwd desde
# el que se lance Kuramoto.py: si lo lanzas desde la raiz del repo,
# acabaras en `<repo>/resultados/...`.
RESULTADOS_BASE = 'resultados'


def crear_carpeta_resultados(subdir, nombre_base):
    """Crea `resultados/<subdir>/<nombre_base>/` evitando sobrescribir.

    Si la carpeta ya existia, anade sufijo (1), (2), ... hasta encontrar
    un nombre libre. Esto permite lanzar la misma config dos veces sin
    miedo a perder la corrida anterior.

    Parametros
    ----------
    subdir : str
        Por ejemplo 'Tarea1', 'Tarea2', 'Conectoma'.
    nombre_base : str
        Identificador legible de la corrida (p.ej. 'N3000_sigmas3_K300_Runs1_t400-1500').

    Devuelve
    --------
    str: ruta ABSOLUTA de la carpeta creada.
    """
    raiz = os.path.join(RESULTADOS_BASE, subdir)
    os.makedirs(raiz, exist_ok=True)

    path = os.path.join(raiz, nombre_base)
    n = 1
    while os.path.exists(path):
        path = os.path.join(raiz, f"{nombre_base}({n})")
        n += 1
    os.makedirs(path)
    return os.path.abspath(path)


def _ruta(directorio, nombre):
    """Asegura que `directorio` existe y devuelve `directorio/nombre`.

    Atajo para pasarselo directo a `fig.savefig(...)` sin tener que
    comprobar antes que la carpeta esta creada."""
    os.makedirs(directorio, exist_ok=True)
    return os.path.join(directorio, nombre)


def guardar_params_txt(run_dir, params_dict):
    """Vuelca un dict de parametros a `run_dir/params.txt` legible.

    Util para reproducir una corrida y para hojear desde el explorador
    de ficheros sin tener que abrir un .npz."""
    with open(os.path.join(run_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write("Parametros de la ejecucion\n")
        f.write("=" * 40 + "\n")
        for k, v in params_dict.items():
            f.write(f"{k:25s} = {v}\n")


# ----------------------------------------------------------------------------
# Logging: duplica stdout/stderr a un fichero log.txt en run_dir
# ----------------------------------------------------------------------------

class Tee:
    """Duplica writes a varios streams (consola + archivo).

    Llamamos a flush() despues de cada write para que el log quede
    grabado aunque el script se caiga a mitad de ejecucion."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # tqdm/joblib preguntan esto para decidir si usar barras de
        # progreso o salida plana. Si CUALQUIER stream es un tty,
        # devolvemos True para conservar la barra en consola.
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)


def iniciar_log(run_dir):
    """Redirige stdout y stderr a un Tee que escribe tambien en run_dir/log.txt.

    Devuelve (log_file, stdout_orig, stderr_orig). Hay que pasarle estos
    tres valores a `cerrar_log` al final."""
    log_path = os.path.join(run_dir, 'log.txt')
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # line-buffered
    log_file.write(f"Log de ejecucion - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.flush()
    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout_orig, log_file)
    sys.stderr = Tee(stderr_orig, log_file)
    return log_file, stdout_orig, stderr_orig


def cerrar_log(log_file, stdout_orig, stderr_orig):
    """Restaura stdout/stderr originales y cierra el archivo de log."""
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    log_file.close()
