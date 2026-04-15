import numpy as np
import matplotlib.pyplot as plt

def simulate_kuramoto_mean_field(N=1000, K=2.0, sigma_omega=1.0, t_max=40, dt=0.1):
    """
    Simula el modelo de Kuramoto y devuelve el parámetro de orden promediado <R>.
    (Ajustamos ligeramente t_max y dt para agilizar las múltiples simulaciones)
    """
    omega = np.random.normal(loc=0.0, scale=sigma_omega, size=N)
    theta = np.random.uniform(-np.pi, np.pi, N)
    
    steps = int(t_max / dt)
    R_history = np.zeros(steps)
    
    for t in range(steps):
        z = np.mean(np.exp(1j * theta))
        R = np.abs(z)
        Psi = np.angle(z)
        
        R_history[t] = R
        dtheta_dt = omega + K * R * np.sin(Psi - theta)
        theta = theta + dtheta_dt * dt
        
    # Promedio de la segunda mitad para evitar el transitorio
    return np.mean(R_history[steps//2:])

if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Definimos los rangos de exploración
    sigma_values = np.linspace(0.5, 3.0, 12) # 12 valores distintos de sigma
    K_values = np.linspace(0.0, 6.0, 60)     # Malla fina de K para detectar bien el salto
    
    Kc_empiricos = []
    
    print("Iniciando cálculo empírico de Kc...")
    print("Esto requiere simular la red completa cientos de veces. Tardará un momento.\n")
    
    # 2. Barrido sobre los valores de sigma
    for sigma in sigma_values:
        R_results = []
        
        # Simulamos la curva <R> vs K para el sigma actual
        for K in K_values:
            r_mean = simulate_kuramoto_mean_field(N=1000, K=K, sigma_omega=sigma)
            R_results.append(r_mean)
            
        # 3. Encontramos el Kc empírico mediante la máxima diferencia
        # np.diff resta el elemento i+1 menos el elemento i
        diferencias = np.diff(R_results)
        
        # np.argmax nos da el índice donde ocurrió el salto más grande
        indice_max_salto = np.argmax(diferencias)
        
        # El valor de Kc corresponde a ese punto de salto
        # (Tomamos el K intermedio entre el punto antes y después del salto para más precisión)
        Kc_empirico = (K_values[indice_max_salto] + K_values[indice_max_salto + 1]) / 2.0
        
        Kc_empiricos.append(Kc_empirico)
        print(f"Sigma: {sigma:.2f} -> Mayor salto detectado en K = {Kc_empirico:.2f}")

    # 4. Calculamos la teoría analítica para comparar
    Kc_teoricos = np.sqrt(8 / np.pi) * sigma_values

    # 5. Visualización
    plt.figure(figsize=(9, 6))
    
    # Curva extraída empíricamente de los datos (CORREGIDO CON r'')
    plt.plot(sigma_values, Kc_empiricos, 'bo-', linewidth=2, markersize=7, 
             label=r'Empírico')
    
    # Línea teórica como referencia
    plt.plot(sigma_values, Kc_teoricos, 'r--', linewidth=2, alpha=0.7, 
             label=r'Teórico ($K_c = \sqrt{8/\pi} \sigma_\omega$)')
    
    plt.title("Cálculo Empírico del Umbral Crítico $K_c$ mediante Diferencias Finitas")
    plt.xlabel(r"Ancho de la distribución de frecuencias ($\sigma_\omega$)")
    plt.ylabel(r"Fuerza de acoplamiento crítica ($K_c$)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()