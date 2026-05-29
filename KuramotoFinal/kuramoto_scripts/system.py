"""
kuramoto_scripts.system
========================

Clase KuramotoSystem: contiene UNA simulacion (una IC, una matriz, un K).

Modo segun la matriz A:
    A is None  -> campo medio  (sim_type 0), integrador O(N).
    A ndarray  -> red          (sim_type 1, 2, 3), integrador O(N^2),
                  con r de orden por nivel.

Niveles (`level_ids`):
    Lista de particiones de los N nodos. Cada particion es un array (N,)
    de enteros 0..n_grupos-1.
        sim_type 1 -> [module_id]                  (1 nivel)
        sim_type 2 -> [module_id, submodule_id]     (2 niveles)
        sim_type 3 -> [hemisferio_id]               (1 nivel)
    Campo medio no usa niveles.

Tras run():
    mean_R, sigma_R   : ⟨R⟩ y su fluctuacion sobre los 2 ULTIMOS bloques.
    mean_rm, sigma_rm : listas (una entrada por nivel); cada entrada es un
                        array (n_grupos,) con el valor por grupo.
    n_steps_used      : pasos efectivamente integrados (parada adaptativa).
"""

import numpy as np
from kuramoto_scripts.integrators import _euler_meanfield, _euler_network


class KuramotoSystem:

    def __init__(self, N, dt, max_steps, block_size, conv_threshold):
        self.N              = N
        self.dt             = dt
        self.max_steps      = max_steps
        self.block_size     = block_size
        self.conv_threshold = conv_threshold

        # Series temporales (se rellenan hasta n_steps_used).
        self.R   = np.zeros(max_steps, dtype=np.float64)
        self.psi = np.zeros(max_steps, dtype=np.float64)

        self.omega     = np.zeros(N, dtype=np.float64)
        self.theta     = np.zeros(N, dtype=np.float64)
        self.theta_new = np.zeros(N, dtype=np.float64)

        # Red y niveles (None en campo medio).
        self.A        = None
        self.n_levels = 0
        self.level_id           = None   # (L, N)
        self.n_groups_per_level = None   # (L,)
        self.group_size         = None   # (L, Gmax)
        self.r_levels           = None   # (L, Gmax, max_steps)

        # Resultados.
        self.n_steps_used = 0
        self.mean_R   = 0.0
        self.sigma_R  = 0.0
        self.mean_rm  = []   # lista de arrays, uno por nivel
        self.sigma_rm = []

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------

    def initialize(self, sigma=1.0, omega=None, theta_0=None,
                   A=None, level_ids=None):
        if omega is None:
            self.omega = np.random.normal(0.0, sigma, self.N)
        else:
            self.omega = np.asarray(omega, dtype=np.float64).copy()

        if theta_0 is None:
            self.theta = np.random.uniform(-np.pi, np.pi, self.N)
        else:
            # .copy() critico: el integrador escribe in-place.
            self.theta = np.asarray(theta_0, dtype=np.float64).copy()

        if A is None:
            self.A        = None
            self.n_levels = 0
            return

        self.A = np.ascontiguousarray(A, dtype=np.float64)

        if not level_ids:
            # Red sin ninguna particion: un unico nivel trivial (todo el grafo).
            level_ids = [np.zeros(self.N, dtype=np.int64)]

        self._montar_niveles(level_ids)

    def _montar_niveles(self, level_ids):
        """Empaqueta la lista de particiones en arrays compactos para Numba."""
        L = len(level_ids)
        n_groups_per_level = np.array(
            [int(np.asarray(lid).max()) + 1 for lid in level_ids], dtype=np.int64)
        Gmax = int(n_groups_per_level.max())

        level_id   = np.zeros((L, self.N), dtype=np.int64)
        group_size = np.zeros((L, Gmax), dtype=np.float64)
        for l, lid in enumerate(level_ids):
            lid = np.asarray(lid, dtype=np.int64)
            level_id[l] = lid
            for g in range(n_groups_per_level[l]):
                group_size[l, g] = float(np.count_nonzero(lid == g))

        self.n_levels           = L
        self.level_id           = level_id
        self.n_groups_per_level = n_groups_per_level
        self.group_size         = group_size
        self.r_levels           = np.zeros((L, Gmax, self.max_steps), dtype=np.float64)

    # ------------------------------------------------------------------
    # Ejecucion
    # ------------------------------------------------------------------

    def run(self, K):
        if self.A is None:
            self.n_steps_used = _euler_meanfield(
                self.theta, self.theta_new, self.omega, K, self.dt,
                self.max_steps, self.block_size, self.conv_threshold,
                self.R, self.psi,
            )
        else:
            rhs = np.zeros(self.N, dtype=np.float64)
            self.n_steps_used = _euler_network(
                self.theta, self.theta_new, self.omega, self.A, K, self.dt,
                self.max_steps, self.block_size, self.conv_threshold,
                self.level_id, self.n_groups_per_level, self.group_size,
                self.R, self.psi, self.r_levels, rhs,
            )
        self._calcular_observables()

    def _ventana_final(self):
        """Indices [lo, hi) de los DOS ultimos bloques (2*block_size valores)."""
        hi = self.n_steps_used + 1
        lo = hi - 2 * self.block_size
        if lo < 0:
            lo = 0
        return lo, hi

    def _calcular_observables(self):
        lo, hi = self._ventana_final()

        ventana_R    = self.R[lo:hi]
        self.mean_R  = float(np.mean(ventana_R))
        self.sigma_R = float(np.std(ventana_R))

        self.mean_rm  = []
        self.sigma_rm = []
        for l in range(self.n_levels):
            ng = int(self.n_groups_per_level[l])
            bloque = self.r_levels[l, :ng, lo:hi]   # (ng, ventana)
            self.mean_rm.append(np.mean(bloque, axis=1))
            self.sigma_rm.append(np.std(bloque, axis=1))


# ----------------------------------------------------------------------------
# Wrapper de conveniencia
# ----------------------------------------------------------------------------

def run_simulation(N, K, sigma, dt, max_steps, block_size, conv_threshold,
                   A=None, level_ids=None, omega=None, theta_0=None):
    """Construye, inicializa y corre UNA simulacion. Devuelve el sistema."""
    sys = KuramotoSystem(N, dt, max_steps, block_size, conv_threshold)
    sys.initialize(sigma=sigma, omega=omega, theta_0=theta_0,
                   A=A, level_ids=level_ids)
    sys.run(K=K)
    return sys
