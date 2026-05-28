"""
kuramoto.system
===============

Clase contenedora KuramotoSystem. Una UNICA simulacion completa (un par
sigma/K, una IC) vive en una instancia de esta clase.

Diseño:

    - Atributo `A` (matriz de adyacencia):
        * None      -> modo campo medio. Llama a _euler_meanfield.
        * ndarray   -> modo red. Llama a _euler_red.

    - Atributo `module_id`:
        * None      -> sin descomposicion por modulos (Tarea 4 conectoma).
                       Internamente se sustituye por un module_id "trivial"
                       (todos los nodos en el modulo 0) y r_m == R.
        * ndarray   -> descomposicion explicita (Tarea 2 modular).
                       r_m[m] tiene sentido fisico.

El usuario no toca _euler_*: solo construye un KuramotoSystem, llama a
initialize(...) y luego run(K). Las propiedades R_mean, R_std, r_m_mean,
r_m_std calculan el estacionario descartando el primer 25% de la serie.
"""

import numpy as np
from kuramoto.integrators import _euler_meanfield, _euler_red


class KuramotoSystem:
    """Contenedor de UNA simulacion.

    Atributos relevantes despues de run(K):
        R[t]    : parametro de orden global, shape (steps+1,)
        psi[t]  : fase media global, shape (steps+1,)
        r_m[m, t]: parametro de orden del modulo m (solo modo red).
                   Si module_id no se paso, r_m existe pero coincide con R.
    """

    def __init__(self, N, steps, dt, num_modules=1):
        self.N           = N           # numero de osciladores
        self.steps       = steps       # numero de pasos temporales
        self.dt          = dt          # paso de tiempo
        self.num_modules = num_modules # >= 1; en campo medio se ignora

        # Series temporales del parametro de orden.
        self.R   = np.zeros(steps + 1, dtype=np.float64)
        self.psi = np.zeros(steps + 1, dtype=np.float64)

        # r_m solo se usa en modo red. En campo medio queda a ceros.
        self.r_m = np.zeros((max(num_modules, 1), steps + 1), dtype=np.float64)

        # Frecuencias naturales (estaticas).
        self.omega = np.zeros(N, dtype=np.float64)

        # Buffers de fases.
        self.theta_curr = np.zeros(N, dtype=np.float64)
        self.theta_next = np.zeros(N, dtype=np.float64)

        # Red. En campo medio quedan None / vacios.
        self.A         = None
        self.module_id = None

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------

    def initialize(self, sigma=1.0, omega=None, theta_0=None,
                   A=None, module_id=None):
        """Carga frecuencias, fases iniciales y (opcional) red.

        Parametros
        ----------
        sigma : float
            Anchura de la Gaussiana usada SOLO si `omega is None`.
        omega : array (N,) o None
            Si se pasa, se usa tal cual (Common Random Numbers).
        theta_0 : array (N,) o None
            Idem. Por defecto uniformes en [-pi, pi].
        A : array (N, N) o None
            None -> campo medio. ndarray -> red.
        module_id : array (N,) o None
            Solo relevante si A != None. Si None, se asume un unico
            modulo (todos los nodos a 0): r_m coincidira con R.
        """
        if omega is None:
            self.omega = np.random.normal(0.0, sigma, self.N)
        else:
            self.omega = np.asarray(omega, dtype=np.float64).copy()

        if theta_0 is None:
            self.theta_curr = np.random.uniform(-np.pi, np.pi, self.N)
        else:
            # .copy() critico: el integrador escribe in-place y no queremos
            # corromper la IC compartida entre workers de joblib.
            self.theta_curr = np.asarray(theta_0, dtype=np.float64).copy()

        if A is None:
            # Modo campo medio.
            self.A         = None
            self.module_id = None
        else:
            self.A = np.ascontiguousarray(A, dtype=np.float64)
            if module_id is None:
                # Sin info de modulos: asignamos todos al modulo 0.
                # r_m[0, :] sera identico a R(t).
                self.module_id   = np.zeros(self.N, dtype=np.int64)
                self.num_modules = 1
                self.r_m         = np.zeros((1, self.steps + 1), dtype=np.float64)
            else:
                self.module_id   = np.asarray(module_id, dtype=np.int64).copy()
                M                = int(self.module_id.max()) + 1
                self.num_modules = M
                self.r_m         = np.zeros((M, self.steps + 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # Ejecucion: dispatch automatico segun haya A o no
    # ------------------------------------------------------------------

    def run(self, K):
        """Integra hasta t = steps*dt con acoplamiento K.

        Despacho automatico:
            A is None -> _euler_meanfield (rapido, O(N) por paso).
            A ndarray -> _euler_red (O(N^2) por paso).
        """
        if self.A is None:
            _euler_meanfield(
                self.theta_curr, self.theta_next, self.omega,
                K, self.dt, self.steps,
                self.R, self.psi,
            )
        else:
            _euler_red(
                self.theta_curr, self.theta_next, self.omega,
                self.A, self.module_id,
                K, self.dt, self.steps,
                self.R, self.psi, self.r_m,
            )

    # ------------------------------------------------------------------
    # Observables del estacionario: descartan el primer 25% (transitorio)
    # ------------------------------------------------------------------

    @property
    def R_mean(self):
        """<R> estacionario."""
        n_trans = self.steps // 4
        return float(np.mean(self.R[n_trans:]))

    @property
    def R_std(self):
        """sigma_R: cuantifica la metaestabilidad (pico en Kc)."""
        n_trans = self.steps // 4
        return float(np.std(self.R[n_trans:]))

    @property
    def r_m_mean(self):
        """<r_m> por modulo. Solo significativo en modo red con module_id."""
        n_trans = self.steps // 4
        return np.mean(self.r_m[:, n_trans:], axis=1)

    @property
    def r_m_std(self):
        """sigma_r_m por modulo."""
        n_trans = self.steps // 4
        return np.std(self.r_m[:, n_trans:], axis=1)


# ----------------------------------------------------------------------------
# Wrapper de conveniencia
# ----------------------------------------------------------------------------

def Simulacion_Kuramoto(N, K, sigma, dt, t_max,
                        A=None, module_id=None,
                        omega=None, theta_0=None):
    """Atajo: construye, inicializa y corre UNA simulacion completa.

    Devuelve el KuramotoSystem para que el llamante extraiga R_mean,
    R_std, r_m_mean, r_m_std, o las series enteras si quiere.
    """
    num_pasos = int(t_max / dt)
    sys = KuramotoSystem(N=N, steps=num_pasos, dt=dt)
    sys.initialize(sigma=sigma, omega=omega, theta_0=theta_0,
                   A=A, module_id=module_id)
    sys.run(K=K)
    return sys
