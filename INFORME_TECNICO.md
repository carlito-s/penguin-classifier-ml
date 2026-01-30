# INFORME TÉCNICO: Sistema de Clasificación Bio-Métrica de Pingüinos (Palmer Archipelago)

**Fecha:** 29 de Enero de 2026

**Autor:** Carlos Daniel López Marín

**Arquitectura:** Random Forest Classifier & Streamlit Interface

---

## 1. Resumen 

El presente proyecto detalla el diseño, desarrollo e implementación de un sistema de aprendizaje automático (*Machine Learning*) capaz de clasificar especies de pingüinos del archipiélago Palmer con una precisión del **100%**. Utilizando un enfoque de **Aprendizaje Supervisado**, se procesaron datos morfométricos brutos para entrenar un algoritmo de **Random Forest**. La solución final se desplegó mediante una interfaz web interactiva desarrollada en Streamlit, optimizada para inferencia en tiempo real y validación visual mediante mapeo de activos locales.

---

## 2. Definición del Problema y Contexto

La identificación precisa de especies animales es crucial para los estudios ecológicos y el monitoreo de la biodiversidad en la Antártida. Los métodos tradicionales requieren observación experta manual, lo cual es costoso y lento.
El objetivo de este sistema es automatizar la identificación de las especies **Adelie**, **Chinstrap** (Barbijo) y **Gentoo** basándose exclusivamente en medidas físicas no invasivas:

* Longitud y profundidad del culmen (pico).
* Longitud de la aleta.
* Masa corporal.
* Sexo e Isla de origen.

**Dataset:** Se utilizó el conjunto de datos *Palmer Penguins* (Gorman, Williams & Fraser, 2014), que consta de 344 registros crudos.

---

## 3. Metodología de Ingeniería de Datos (Data Engineering)

Antes de alimentar el modelo, los datos fueron sometidos a un pipeline de limpieza y transformación en `src/data_processing.py`.

### 3.1 Tratamiento de Valores Ausentes (Imputación)

El dataset original contenía valores nulos (NaN). Eliminarlos reduciría la muestra estadística significativamente. Se optó por **Imputación Estadística**:

* **Variables Numéricas:** Se utilizó la **Mediana**. A diferencia de la media, la mediana es robusta y no se ve sesgada por *outliers* (valores atípicos extremos) biológicos.
* **Variables Categóricas:** Se utilizó la **Moda** (valor más frecuente) para preservar la distribución de probabilidad de las clases.

### 3.2 Codificación de Variables (Encoding)

Los algoritmos de ML requieren entradas numéricas. Se aplicaron dos técnicas distintas:

1. **Label Encoding (Target):** Para la variable objetivo (`species`).
* *Mapping:* Adelie  0, Chinstrap  1, Gentoo  2.


2. **One-Hot Encoding (Features):** Para `island` y `sex`.
* *Justificación:* Asignar números (0, 1, 2) a las islas implicaría una relación de orden falsa (ej: Isla Biscoe > Isla Dream), lo cual confundiría al modelo. *One-Hot* crea columnas binarias independientes (ej: `island_Dream`: 0 o 1).



---

## 4. Marco Teórico del Algoritmo: Random Forest

El núcleo del sistema es un modelo de **Random Forest (Bosque Aleatorio)**. Para comprender su eficacia, debemos desglosar su funcionamiento interno.

### 4.1 Concepto de "Ensemble Learning" (Aprendizaje en Conjunto)

Random Forest no es un solo modelo, es un conjunto (*ensemble*) de múltiples **Árboles de Decisión**. Se basa en la teoría de la "Sabiduría de las Masas": muchos modelos débiles trabajando juntos (y votando) superan el rendimiento de un solo modelo complejo propensa al error.

### 4.2 Mecanismo de Funcionamiento

El algoritmo sigue el método de **Bagging (Bootstrap Aggregating)**:

1. **Bootstrapping (Muestreo):** Si tenemos 100 árboles, cada árbol se entrena con una sub-muestra aleatoria de los datos originales, permitiendo repeticiones. Esto garantiza que cada árbol vea un escenario ligeramente diferente.
2. **Aleatoriedad de Características:** En cada nodo de decisión de un árbol, el algoritmo no busca la mejor división entre *todas* las variables, sino entre un subconjunto aleatorio de ellas. Esto "descorrelaciona" los árboles, evitando que todos cometan el mismo error.
3. **Votación (Majority Voting):** Para clasificar un pingüino nuevo, cada uno de los 100 árboles emite un voto. La clase con la mayoría de votos gana.

### 4.3 Matemáticas de la Decisión (Impureza de Gini)

Cada árbol decide cómo dividir los datos (ej: "¿Pico > 45mm?") buscando minimizar la **Impureza de Gini**.
La fórmula de Gini para un nodo  es:

$$Gini(t) = 1 - \sum_{i=0}^{c} (p_i)^2$$

Donde  es la probabilidad de una clase  en ese nodo.

* **Gini = 0:** Pureza total (todos los pingüinos en el nodo son de la misma especie).
* El algoritmo busca cortes que reduzcan el Gini lo más rápido posible.

---

## 5. Arquitectura del Modelo y Configuración

El modelo fue instanciado utilizando la librería `scikit-learn` con la siguiente configuración:

| Hiperparámetro | Valor | Justificación Técnica |
| --- | --- | --- |
| **Algoritmo** | `RandomForestClassifier` | Maneja excelente la no-linealidad y el ruido. |
| **n_estimators** | `100` | Cantidad de árboles. 100 ofrece un equilibrio entre precisión y costo computacional. |
| **random_state** | `42` | Semilla para garantizar la **reproducibilidad** científica de los resultados. |
| **Split Train/Test** | `80% / 20%` | Estándar de la industria (Regla de Pareto) para evitar *Overfitting*. |

---

## 6. Análisis de Resultados y Desempeño

El modelo fue evaluado con el conjunto de prueba (69 pingüinos que el modelo nunca vio durante el entrenamiento).

### 6.1 Matriz de Confusión y Métricas

El modelo alcanzó un desempeño perfecto en este conjunto de prueba.

* **Accuracy (Exactitud Global):** **1.00 (100%)**
* **F1-Score:** **1.00** para todas las clases.

| Especie | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Adelie | 1.00 | 1.00 | 1.00 | 32 |
| Chinstrap | 1.00 | 1.00 | 1.00 | 16 |
| Gentoo | 1.00 | 1.00 | 1.00 | 21 |

> **Interpretación:** La precisión del 100% sugiere que las especies de pingüinos en este dataset son **linealmente separables** en el hiperespacio multidimensional definido por sus medidas. Las diferencias biológicas son tan marcadas que el modelo no tiene ambigüedad.

### 6.2 Importancia de las Características (Feature Importance)

Utilizando la propiedad `feature_importances_` del Random Forest, determinamos qué variables biológicas son las más determinantes para la clasificación:

1. **Longitud del Pico (Bill Length):** Es la variable discriminante principal (separa Adelie de Chinstrap).
2. **Longitud de la Aleta (Flipper Length):** Variable secundaria crítica (separa Gentoo del resto por su gran tamaño).
3. **Profundidad del Pico:** Ayuda a refinar la distinción entre sexos y especies similares.

---

## 7. Implementación de Software y UX

* **Backend:** Refactorizado en `src/train.py` para asegurar la persistencia atómica del modelo (`.pkl`) y el codificador.
* **Frontend:** Aplicación desarrollada en **Streamlit** (`app.py`).
* *Optimización:* Uso de decoradores `@st.cache_resource` para cargar el modelo en memoria una sola vez, reduciendo la latencia de inferencia a <50ms.
* *Feedback Visual:* Implementación de carga de imágenes locales (`.webp`) mapeadas dinámicamente al resultado de la predicción, mejorando la experiencia de usuario y reduciendo la dependencia de red.



---

## 8. Conclusiones y Trabajo Futuro

El sistema ha demostrado ser una herramienta robusta y altamente precisa para la clasificación taxonómica automatizada. La arquitectura basada en Random Forest probó ser superior para este tipo de datos tabulares biológicos.

**Próximos Pasos (Roadmap):**

1. **Contenerización:** Crear un `Dockerfile` para desplegar la aplicación en cualquier servidor en la nube (AWS/Azure).
2. **API REST:** Desacoplar la predicción en una API con **FastAPI** para permitir que otros sistemas (ej: drones o cámaras inteligentes) consuman el modelo.
3. **Expansión del Dataset:** Incorporar datos de nuevas temporadas o incluir nuevas especies antárticas (ej: Pingüino Macaroni) para probar la capacidad de generalización del modelo.


