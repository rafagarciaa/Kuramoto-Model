"""
Paquete kuramoto
================

Implementacion del modelo de Kuramoto:

    theta_dot_i = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)

con dos modos automaticos segun la matriz de adyacencia A:

    A is None  -> campo medio (Tarea 1).
                  Integrador rapido O(N) por paso, malla de K LINEAL en
                  torno al Kc teorico, observable global R(t).

    A ndarray  -> red arbitraria (Tareas 2, 3, 4).
                  Integrador O(N^2) por paso, malla de K LOGARITMICA,
                  observable global R(t) y opcionalmente r_m(t) por
                  modulo si se pasa `module_id`.

El usuario solo tiene que rellenar parametros en `Kuramoto.py` y elegir
la matriz `MatrixOp`. Toda la maquinaria (barrido, ICs pareadas, joblib,
plots, logs, etc.) vive aqui dentro.
"""

from kuramoto.system        import KuramotoSystem, Simulacion_Kuramoto
from kuramoto.observables   import Kc_teorica, Kc_experimental
from kuramoto.grids         import (
    K_values_tstudent, K_values_log_tstudent,
    t_max_per_K, t_max_per_K_log,
)
from kuramoto.networks      import (
    generar_ICs, generar_As_modular, crear_matriz_modular,
    stats_matriz_adyacencia,
)
from kuramoto.conectoma     import (
    cargar_conectoma_promedio, cargar_conectoma_sujetos,
    binarizar, es_conexo, buscar_threshold_optimo,
    cargar_y_preparar_A,
)
from kuramoto.sweep         import barrido_completo
from kuramoto.io            import (
    crear_carpeta_resultados, guardar_params_txt,
    iniciar_log, cerrar_log, Tee, _ruta,
)
from kuramoto.plotting      import (
    setup_plot_style,
    plot_R_vs_K, plot_sigmaR_vs_K, plot_combined,
    plot_matriz_adyacencia,
)
