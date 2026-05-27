import numpy as np
from matplotlib.pyplot import * 
from math import * 
from dataclasses import dataclass




@dataclass
class KuramotoSystem:

    N: int 
    A: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    K:  float
    dt: float
    t_max: float
    n_steps: int
    R_t: np.ndarray

def Inicializacion(N, A, K, dt, t_max, sigma, Dist_Is_Gaussian, sigma_center):

    n_steps = int(t_max/dt) + 1
    
    theta = np.zeros((N, n_steps))
    theta[:, 0] = np.random.uniform(- pi, pi, N)
    if Dist_Is_Gaussian == True:
        omega = np.random.normal(sigma_center, sigma, N)
        omega = sigma_center + sigma * np.random.standard_cauchy(N)
    R_t = np.zeros(n_steps)
    system = KuramotoSystem(N=N, A=A, theta=theta, omega=omega, K=K, dt=dt, t_max=t_max, n_steps=n_steps, R_t=R_t)

    return system

def IntegracionEuler(system: KuramotoSystem, step):
    """Esto es una función recursiva, que calcula todos los thetas de los pasos de tiempo del sistema de kuramoto hasta llegar a t_max.
    
    Args:
        system (KuramotoSystem): Sistema de Kuramoto
        step (int): Paso de tiempo inicial, necesario por la recursividad.
    
    Returns:
        KuramotoSystem: Sistema de Kuramoto actualizado con los thetas calculados hasta el paso de tiempo t_max.
    """
    t = step * system.dt

    if t < system.t_max:
        for i in range(system.N):
            suma = 0
            for j in range(system.N):
                suma += system.A * sin(system.theta[j] - system.theta[i])
            system.theta[i, step + 1] = system.theta[i, step] + system.dt * (system.omega[i] + system.K * suma)
        return IntegracionEuler(system, step + 1)
    
    else: 
        system.R_t = R_t(system)
        return system

def R_t(system: KuramotoSystem):
    
    suma = np.zeros(system.n_steps, dtype=complex)
    for i in range(system.N):
        suma += np.exp(1j * system.theta[i, :]) 
    system.R_t = abs(suma) / system.N
    
    return system

def SimulacionKuramoto(K, t_max, dt, N, A, sigma):

    return IntegracionEuler(Inicializacion(N, A, K, dt, t_max, sigma), 0)

def KuramotoRuns(system: KuramotoSystem, num_runs):

def main():
    
    return


if __name__ == "__main__":
    main()