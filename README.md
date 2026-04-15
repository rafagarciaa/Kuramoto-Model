# Modelo de Kuramoto — Sistemas Complejos

Proyecto académico sobre el **modelo de Kuramoto** para la asignatura de Sistemas Complejos (5.º curso, 2.º cuatrimestre).

## Descripción

Este repositorio contiene las simulaciones numéricas y el análisis teórico del modelo de Kuramoto, incluyendo:

- **Transición a la sincronización**: Estudio de la transición de fase en función de la fuerza de acoplamiento $K$ para diferentes distribuciones de frecuencias intrínsecas.
- **Metaestabilidad y sincronización jerárquica**: Análisis en redes modulares y con estructura de tipo cerebral (*brain-like networks*).
- **Cálculo empírico del umbral crítico** $K_c$: Comparación entre resultados numéricos y predicción analítica $K_c = \sqrt{8/\pi}\,\sigma_\omega$.

## Estructura del Proyecto

```
Kuramoto/
├── src/                    # Código fuente de las simulaciones
│   ├── kuramoto1.py        # Transición de sincronización (R vs K)
│   └── kuramoto2.py        # Cálculo empírico de Kc(σ)
├── figures/                # Figuras generadas por las simulaciones
├── data/                   # Datos de salida de las simulaciones
├── report/                 # Informe final (LaTeX)
├── bibliography/           # Artículos de referencia
├── notebooks/              # Jupyter notebooks (exploración)
├── requirements.txt        # Dependencias de Python
└── README.md
```

## Requisitos

```bash
pip install numpy matplotlib
```

## Uso

```bash
# Simulación 1: Transición de fase R(K) para diferentes σ
python src/kuramoto1.py

# Simulación 2: Cálculo empírico de Kc en función de σ
python src/kuramoto2.py
```

## Resultados Principales

- Se observa la transición de fase de segundo orden predicha por la teoría de Kuramoto.
- El umbral crítico empírico $K_c$ concuerda con la predicción analítica.
- Se analiza metaestabilidad en redes con estructura modular.

## Autores

- Rafael (Universidad)

## Licencia

Proyecto académico — Uso educativo.
