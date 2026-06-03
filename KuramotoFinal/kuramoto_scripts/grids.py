"""
kuramoto_scripts.grids
======================

Colocadores de puntos K para el barrido.

Filosofia: donde la curva R(K) tiene pendiente (alrededor de Kc) hacen
falta MAS puntos para resolver bien la transicion; en las colas (K << Kc
o K >> Kc) bastan pocos. Para lograrlo usamos una densidad t-Student
centrada en Kc, con pico en Kc y colas suaves.

(El tiempo de simulacion por punto ya NO se reparte aqui: ahora cada
simulacion se para sola por convergencia, ver integrators.py.)

Dos variantes:
    - K_values_tstudent     : densidad t-Student en K        (campo medio).
    - K_values_log_tstudent : densidad t-Student en log(K)    (red, varios
                              ordenes de magnitud en K).
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
    u_min = t_dist.cdf(K_min, df=df, loc=1.05 * Kc, scale=sigma_K)
    u_max = t_dist.cdf(K_max, df=df, loc=1.05 * Kc, scale=sigma_K)

    u_values = np.linspace(u_min, u_max, num_K)

    return t_dist.ppf(u_values, df=df, loc=Kc, scale=sigma_K)

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

