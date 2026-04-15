import numpy as np
import matplotlib.pyplot as plt

def simulate_kuramoto_mean_field(N=1000, K=2.0, sigma_omega=1.0, t_max=50, dt=0.05):
    """
    Simula el modelo de Kuramoto clásico (campo medio) usando el método de Euler.
    Devuelve el parámetro de orden promediado en el tiempo <R>.
    """
    # 1. Inicialización
    # Frecuencias intrínsecas: Distribución Gaussiana centrada en 0
    omega = np.random.normal(loc=0.0, scale=sigma_omega, size=N)
    
    # Fases iniciales: Aleatorias entre -pi y pi
    theta = np.random.uniform(-np.pi, np.pi, N)
    
    steps = int(t_max / dt)
    R_history = np.zeros(steps)
    
    # 2. Bucle temporal (Método de Euler)
    for t in range(steps):
        # Calcular el parámetro de orden macroscópico
        z = np.mean(np.exp(1j * theta))
        R = np.abs(z)
        Psi = np.angle(z)
        
        R_history[t] = R
        
        # Calcular la derivada con la ecuación de campo medio optimizada
        dtheta_dt = omega + K * R * np.sin(Psi - theta)
        
        # Método de Euler: actualizar fases
        theta = theta + dtheta_dt * dt
        
    # 3. Promediamos R ignorando la primera mitad de la simulación (estado transitorio)
    # para asegurar que el sistema se ha estabilizado
    R_mean = np.mean(R_history[steps//2:])
    
    return R_mean

if __name__ == "__main__":
    # Fijamos la semilla para que los resultados sean reproducibles
    np.random.seed(42) 
    
    # Parámetros del experimento
    K_values = np.linspace(0.0, 8.0, 40)
    sigmas = [0.5, 1.0, 2.0] # Diferentes anchuras de distribución de frecuencias
    
    plt.figure(figsize=(10, 6))
    print("Calculando las transiciones. Esto puede tardar un par de minutos...")
    
    # Bucle sobre las diferentes dispersiones
    for sigma in sigmas:
        print(f"Simulando para sigma_omega = {sigma}...")
        R_results = []
        
        # Bucle sobre las fuerzas de acoplamiento
        for K in K_values:
            r_mean = simulate_kuramoto_mean_field(N=1000, K=K, sigma_omega=sigma, t_max=50, dt=0.05)
            R_results.append(r_mean)
        
        # Valor teórico de Kc para una distribución Gaussiana
        Kc_teorico = np.sqrt(8/np.pi) * sigma
        
        # Añadir la curva al gráfico
        plt.plot(K_values, R_results, 'o-', markersize=4, 
                 label=rf'$\sigma_\omega = {sigma}$ ($K_c \approx {Kc_teorico:.2f}$)')
        
        # Dibujar una línea vertical en el Kc teórico
        plt.axvline(x=Kc_teorico, linestyle='--', alpha=0.5)

    # Configuración final del gráfico
    plt.title("Transición a la Sincronización en el Modelo de Kuramoto (Campo Medio)")
    plt.xlabel("Fuerza de acoplamiento (K)")
    plt.ylabel(r"Coherencia Global $\langle R \rangle$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("¡Simulación terminada!")