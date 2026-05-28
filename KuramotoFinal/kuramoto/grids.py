"""
kuramoto.grids
==============

Colocadores de puntos K y de tiempos de simulacion t_max(K).

Filosofia general:

    En las regiones donde la curva R(K) tiene pendiente (alrededor de Kc),
    necesitamos:
        - MAS puntos de K para resolver bien la transicion.
        - MAS tiempo de simulacion en cada K, porque ahi sigma_R es alto
          y necesitamos muestrear mas para estimar bien <R>.

    En las colas (K << Kc o K >> Kc) ocurre lo opuesto: pocos puntos y
    poco tiempo bastan.

    Para lograrlo usamos una densidad t-Student centrada en Kc, que tiene
    el pico en Kc y colas suaves. La MISMA densidad sirve para colocar
    los K y para asignar t_max(K), asi que (1) la concentracion de puntos
    y (2) la concentracion de potencia de computo van de la mano.

Dos variantes:

    Lineal (campo medio, Tarea 1):
        - K_values_tstudent      : densidad t-Student en K.
        - t_max_per_K            : t_max(K) con la misma forma.

    Logaritmica (red, Tareas 2-4):
        - K_values_log_tstudent  : densidad t-Student en log(K). Necesaria
          cuando el rango util cubre varios ordenes de magnitud.
        - t_max_per_K_log        : t_max(K) con la misma forma en log(K).
"""

import numpy as np
from scipy.stats import t as t_dist


# ----------------------------------------------------------------------------
# Variante LINEAL (Tarea 1 / campo medio)
# ----------------------------------------------------------------------------

def K_values_tstudent(num_K, K_min, K_max, Kc, width_factor=0.3, df=2):
    """Devuelve `num_K` valores de K con densidad t-Student centrada en Kc.

    La densidad t (df=2) es simetrica y tiene colas mas pesadas que la
    Gaussiana, asi que mantiene resolucion razonable lejos de Kc sin
    desperdiciar puntos en la zona constante (R~0 o R~R_inf).

    Parametros
    ----------
    num_K : int
    K_min, K_max : float
        Limites del barrido.
    Kc : float
        Centro de la densidad (Kc teorico o estimacion previa).
    width_factor : float
        Anchura como fraccion de Kc. 0.3 funciona bien para Gaussiana.
    df : int
        Grados de libertad. df=2 -> colas pesadas; df>30 ~ Gaussiana.
    """
    sigma_K = width_factor * Kc

    # Invertimos la CDF: uniforme en [u_min, u_max] -> K no uniforme.
    u_min = t_dist.cdf(K_min, df=df, loc=Kc, scale=sigma_K)
    u_max = t_dist.cdf(K_max, df=df, loc=Kc, scale=sigma_K)

    u_values = np.linspace(u_min, u_max, num_K)
    return t_dist.ppf(u_values, df=df, loc=Kc, scale=sigma_K)


def t_max_per_K(K_values, Kc, t_max_base, t_max_peak, width_factor=0.3, df=2):
    """Asigna t_max(K) con la MISMA forma t-Student que K_values_tstudent.

        T(K) = T_base + (T_peak - T_base) * pdf_t(K) / pdf_t(Kc)

    Asi un unico (width_factor, df) controla simultaneamente DONDE colocas
    los K y CUANTO tiempo se simula cada K.
    """
    sigma_K = width_factor * Kc
    pdf     = t_dist.pdf(K_values, df=df, loc=Kc, scale=sigma_K)
    pdf_max = t_dist.pdf(Kc,        df=df, loc=Kc, scale=sigma_K)
    weight  = pdf / pdf_max
    return t_max_base + (t_max_peak - t_max_base) * weight


# ----------------------------------------------------------------------------
# Variante LOGARITMICA (Tareas 2-4 / red)
# ----------------------------------------------------------------------------

def K_values_log_tstudent(num_K, K_min, K_max, K_center=None, width_factor=0.5, df=2):
    """Devuelve `num_K` valores de K con densidad t-Student en log(K).

    Cuando `K_center is None`, cae a `np.geomspace` (logaritmico uniforme),
    util cuando todavia no sabemos donde esperar la transicion.

    Cuando `K_center` esta dado, concentra puntos en log(K_center), util
    una vez identificado el Kc aproximado.
    """
    log_K_min = np.log(K_min)
    log_K_max = np.log(K_max)

    if K_center is None:
        return np.exp(np.linspace(log_K_min, log_K_max, num_K))

    log_K_center = np.log(K_center)
    width        = width_factor * max(abs(log_K_center), 1.0)

    q_min = t_dist.cdf(log_K_min, df=df, loc=log_K_center, scale=width)
    q_max = t_dist.cdf(log_K_max, df=df, loc=log_K_center, scale=width)
    qs    = np.linspace(q_min, q_max, num_K)
    log_K = t_dist.ppf(qs, df=df, loc=log_K_center, scale=width)
    return np.exp(log_K)


def t_max_per_K_log(K_values, K_center, t_max_base, t_max_peak, width_factor=0.5, df=2):
    """Analogo a `t_max_per_K` pero con la forma en log(K).

    Si K_center es None, devuelve un array constante a t_max_base
    (no sabemos donde concentrar tiempo). Si K_center esta dado,
    da pico en K_center con la misma forma t-Student que K_values_log_tstudent.
    """
    K_values = np.asarray(K_values, dtype=np.float64)
    if K_center is None:
        return np.full_like(K_values, t_max_base)

    log_K        = np.log(K_values)
    log_K_center = np.log(K_center)
    width        = width_factor * max(abs(log_K_center), 1.0)

    pdf     = t_dist.pdf(log_K,        df=df, loc=log_K_center, scale=width)
    pdf_max = t_dist.pdf(log_K_center, df=df, loc=log_K_center, scale=width)
    weight  = pdf / pdf_max
    return t_max_base + (t_max_peak - t_max_base) * weight
