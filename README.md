# Clasificador de Especies de Pingüinos 🐧
Proyecto de Machine Learning utilizando el dataset *Palmer Penguins* para clasificar especies mediante un modelo de **Random Forest**.

## Requerimientos del Proyecto
- **Lenguaje:** Python 3.9+
- **Librerías principales:** Scikit-Learn, Pandas, Seaborn.
- **Objetivo:** Alcanzar un F1-Score > 0.85.

## Estructura del Proyecto
- `notebooks/`: Análisis Exploratorio de Datos (EDA).

- `src/`: Scripts de procesamiento y modelado.

## 📊 Evaluación del Modelo y Resultados

El modelo fue evaluado utilizando un conjunto de datos independiente (20% del total) que el algoritmo nunca vio durante el entrenamiento. Los resultados demuestran una alta fiabilidad en la clasificación.

### 📈 Métricas de Desempeño
| Especie | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Adelie** | 0.94 | 0.94 | 0.94 |
| **Chinstrap** | 0.92 | 0.92 | 0.92 |
| **Gentoo** | 1.00 | 1.00 | 1.00 |
| **Promedio Global** | **0.96** | **0.96** | **0.96** |

> **Nota del Arquitecto:** El modelo logra un F1-Score perfecto (1.00) en la especie **Gentoo**, lo cual es consistente con nuestro EDA inicial, donde observamos que sus características físicas (masa y aletas) lo separan claramente de las otras especies.

### 🧠 Importancia de las Características (Explicabilidad)
Utilizando la técnica de *Gini Importance* del Random Forest, determinamos qué factores físicos son determinantes para la IA:

1. **Longitud del Pico (Bill Length):** ~40% de influencia.
2. **Longitud de la Aleta (Flipper Length):** ~30% de influencia.
3. **Profundidad del Pico (Bill Depth):** ~15% de influencia.
4. **Masa Corporal y Localización:** ~15% restante.



### 🧩 Matriz de Confusión
La matriz de confusión revela que las mínimas confusiones del modelo ocurren entre las especies **Adelie** y **Chinstrap**, debido a sus similitudes morfológicas en ciertas islas compartidas.