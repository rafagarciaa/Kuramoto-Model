"""
kuramoto.observables
====================

Estimadores de Kc (acoplamiento critico) a partir de un barrido.

    - Kc_teorica(sigma):
        Valor exacto de Kuramoto en campo medio para distribucion
        Gaussiana de frecuencias intrinsecas.

    - Kc_experimental(K_values, R_stds, log=False):
        Kc estimado por ajuste parabolico al maximo de sigma_R(K).
        Si log=True, el ajuste se hace en log(K) — apropiado cuando la
        rejilla de K es logaritmica (caso de red, Tareas 2-4).
"""

import math
import numpy as np


def Kc_teorica(sigma):
    """Kc exacto en campo medio para g(omega) Gaussiana de anchura `sigma`:

        Kc = sigma * sqrt(8 / pi)

    Solo tiene sentido en campo medio (Tarea 1). En red arbitraria el Kc
    depende de la estructura de A y no hay formula cerrada general.
    """
    return sigma * math.sqrt(8.0 / math.pi)


def Kc_experimental(K_values, R_stds, window=3, log=False):
    """Estima Kc como el maximo de sigma_R(K) afinado por una parabola.

    Justificacion:
        Tomar simplemente argmax(sigma_R) ata el resultado a la rejilla
        discreta de K. Si en cambio ajustamos una parabola a los
        (2*window + 1) puntos centrados en el maximo, el vertice analitico
        nos da un Kc CONTINUO, no atado a la rejilla, y reduce el ruido
        local del estimador.

    Salvaguardas:
        - Si la ventana no llega a 3 puntos, devuelve argmax discreto.
        - Si la parabola no es concava (a >= 0), devuelve argmax discreto.
        - Si el vertice cae fuera de la ventana, devuelve argmax discreto.

    Parametros
    ----------
    K_values : array_like
        Valores de K del barrido.
    R_stds : array_like
        sigma_R(K) correspondiente a cada K.
    window : int
        Puntos a cada lado del maximo a usar en el ajuste.
        Total ajustados = 2*window + 1.
    log : bool
        Si True, ajusta en log(K). Usar cuando la rejilla de K es log.

    Devuelve
    --------
    float : estimacion continua de Kc.
    """
    K_values = np.asarray(K_values)
    R_stds   = np.asarray(R_stds)
    idx      = int(np.argmax(R_stds))

    lo = max(0, idx - window)
    hi = min(len(K_values), idx + window + 1)

    if hi - lo < 3:
        return float(K_values[idx])

    K_win = K_values[lo:hi]
    R_win = R_stds[lo:hi]

    x = np.log(K_win) if log else K_win

    # y = a*x^2 + b*x + c
    a, b, _ = np.polyfit(x, R_win, 2)

    # Sin concavidad hacia abajo no hay vertice util.
    if a >= 0:
        return float(K_values[idx])

    x_vertex = -b / (2.0 * a)

    # Si el vertice escapa de la ventana, el ajuste no es fiable.
    if x_vertex < x[0] or x_vertex > x[-1]:
        return float(K_values[idx])

    return float(np.exp(x_vertex)) if log else float(x_vertex)
