"""
Paquete kuramoto_scripts
========================

Modelo de Kuramoto:  theta_dot_i = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)

Cuatro tipos de simulacion (variable sim_type en params.json):

    0 -> campo medio.        Barrido en sigma, K lineal. Solo R global.
    1 -> red modular.        n_modules bloques simetricos. R + r_m por modulo.
    2 -> red jerarquica.     2 niveles. R + r^1 (modulos) + r^2 (submodulos).
    3 -> conectoma (.mat).   R + r por hemisferio, y comparacion de la
                             metaestabilidad contra una red aleatoria que
                             preserva el grado.

Parada adaptativa: cada `block_size` pasos se promedia R; si dos bloques
consecutivos difieren menos que `conv_threshold`, la simulacion para. El
⟨R⟩ y sigma_R salen de los dos ultimos bloques.

Todos los parametros viven en params.json.
"""

from kuramoto_scripts.config       import load_params, Params
from kuramoto_scripts.system       import KuramotoSystem, run_simulation
from kuramoto_scripts.observables  import Kc_teorica, Kc_experimental
from kuramoto_scripts.grids        import K_values_tstudent, K_values_log_tstudent
from kuramoto_scripts.networks     import (
    generar_ICs, generar_ICs_por_sigma,
    crear_matriz_modular, generar_As_modular,
    crear_matriz_jerarquica, generar_As_jerarquica,
    stats_matriz_adyacencia,
)
from kuramoto_scripts.conectoma    import (
    cargar_conectoma_promedio, cargar_conectoma_sujetos,
    binarizar, es_conexo, buscar_threshold_optimo,
    cargar_y_preparar_A, hemisferio_ids, randomize_preserving_degree,
)
from kuramoto_scripts.sweep        import barrido
from kuramoto_scripts.io           import (
    crear_carpeta_resultados, guardar_params_txt,
    iniciar_log, cerrar_log, Tee, _ruta,
)
from kuramoto_scripts.plotting     import (
    setup_plot_style,
    plot_mean_field, plot_modular, plot_hierarchical, plot_connectome,
    plot_scaling_Kc, plot_matriz_adyacencia,
)
