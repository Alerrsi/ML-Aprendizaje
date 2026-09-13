# 🧠 Machine Learning & Deep Learning Lab

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Farama-000000?logo=openai&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Activo-success)

**Repositorio de aprendizaje práctico, experimentación e implementación de algoritmos de Inteligencia Artificial: desde modelos clásicos de regresión hasta Redes Neuronales Convolucionales (CNN), Autoencoders y Aprendizaje por Refuerzo Profundo (DRL).**

[Explorar Módulos](#-módulos-del-proyecto) • [Instalación](#-instalación-y-requisitos) • [Estructura](#-estructura-del-repositorio) • [Tecnologías](#-stack-tecnológico)

</div>

---

## 📌 Tabla de Contenidos

- [🎯 Descripción General](#-descripción-general)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [🔬 Módulos del Proyecto](#-módulos-del-proyecto)
  - [1. Regresión Lineal & Predicción](#1-regresión-lineal--predicción)
  - [2. Redes Neuronales & Visión por Computadora](#2-redes-neuronales--visión-por-computadora)
  - [3. Deep Reinforcement Learning (DRL)](#3-deep-reinforcement-learning-drl)
- [🛠️ Stack Tecnológico](#️-stack-tecnológico)
- [🚀 Instalación y Requisitos](#-instalación-y-requisitos)
- [💻 Guía de Ejecución](#-guía-de-ejecución)
- [👥 Autores](#-autores)

---

## 🎯 Descripción General

Este repositorio funciona como un **laboratorio integral de Machine Learning**, documentando la transición desde fundamentos teóricos y modelos estadísticos básicos hasta arquitecturas avanzadas de Deep Learning y agentes autónomos:

* **Modelos Supervisados Clásicos**: Regresión Lineal múltiple aplicada a economía deportiva y algoritmos de clasificación como K-Nearest Neighbors (KNN).
* **Deep Learning & Computer Vision**: Redes neuronales perceptrón multicapa (MLP) y redes convolucionales (CNN) usando **Keras 3** sobre backend de **PyTorch**.
* **Modelos Auto-Supervisados**: Compresión y reconstrucción de imágenes vía Autoencoders con espacio latente.
* **Aprendizaje por Refuerzo (RL)**: Agentes entrenados en entornos de **Gymnasium** (*FrozenLake-v1*) mediante algoritmos de Q-Learning tabular y aproximación de funciones Q con redes neuronales (DQN).

---

## 📂 Estructura del Repositorio

```bash
ML-Aprendizaje/
├── DRL/                               # Aprendizaje por Refuerzo Profundo
│   ├── guia1.py                       # Q-Learning Tabular en FrozenLake-v1
│   ├── guia2.py                       # Deep Q-Learning (DQN) con MLPRegressor
│   └── guia_aprendizaje_por_refuerzo.pdf # Guía teórica y fundamentos de RL
│
├── Linear Regressión/                 # Modelos de Regresión Clásica
│   ├── Decenso gradiente.txt          # Apuntes conceptuales de optimización
│   ├── Función de perdida.txt         # Métricas de error (MSE, RMSE, etc.)
│   └── Premier league/
│       ├── final_data.csv             # Dataset de jugadores y atributos
│       └── main.py                    # Modelo de predicción de valor de mercado
│
├── Redes neuronales/                  # Redes Neuronales y Visión Artificial
│   ├── dataset.py                     # Carga y visualización de dígitos (load_digits)
│   ├── knn_digits.py                  # Clasificación con K-Nearest Neighbors
│   ├── clasificar_digitos.py          # Clasificación con MLPClassifier + análisis de error
│   ├── autoencoder_digits_etiquetado.py # Autoencoder (64 -> 32 -> 16 -> 32 -> 64)
│   └── CNN/
│       ├── Plantilla_ES1_Machine_Learning.ipynb # Notebook interactivo (Fashion-MNIST)
│       └── codigo_completo.py         # Pipeline CNN (Keras + PyTorch)
│
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Documentación principal
```

---

## 🔬 Módulos del Proyecto

### 1. Regresión Lineal & Predicción

Ubicación: `Linear Regressión/Premier league/`

* **Objetivo**: Estimar el valor de mercado (`current_value`) de futbolistas de la Premier League a partir de sus estadísticas de rendimiento, atributos físicos y edad.
* **Técnicas**:
  * Limpieza de datos y selección de características numéricas con `pandas`.
  * Partición de datos con `train_test_split` (80% entrenamiento / 20% test).
  * Ajuste mediante `LinearRegression` de `scikit-learn`.
  * Evaluación mediante el cálculo de la Raíz del Error Cuadrático Medio (RMSE).

```python
# Ejecución rápida
python "Linear Regressión/Premier league/main.py"
```

---

### 2. Redes Neuronales & Visión por Computadora

Ubicación: `Redes neuronales/`

#### 🔢 Clasificación de Dígitos Manuscritos (`load_digits`)
* **`dataset.py`**: Inspección y renderizado de dígitos en escala de grises ($8 \times 8$ píxeles).
* **`knn_digits.py`**: Clasificación basada en distancias euclidianas con $K=5$, alcanzando alta precisión basal.
* **`clasificar_digitos.py`**: Perceptrón Multicapa (`MLPClassifier`) con capa oculta de 64 neuronas:
  * Matriz de confusión interactiva con `ConfusionMatrixDisplay`.
  * Visualización específica de casos donde el modelo falló para diagnóstico cualitativo.

#### 🔄 Autoencoders y Espacio Latente (`autoencoder_digits_etiquetado.py`)
* **Paradigma**: Aprendizaje auto-supervisado (la imagen original actúa como entrada y objetivo: $X \to X$).
* **Arquitectura de Cuello de Botella (Bottleneck)**:
  $$\text{Entrada (64)} \longrightarrow \text{32} \longrightarrow \mathbf{Espacio\ Latente\ (16)} \longrightarrow \text{32} \longrightarrow \text{Reconstrucción (64)}$$
* **Resultados**: Permite evaluar cómo la red comprime patrones visuales esenciales y los reconstruye minimizando el error cuadrático medio (`MSE`).

#### 👕 Clasificación con Redes Convolucionales (CNN) (`Redes neuronales/CNN/`)
* **Dataset**: *Fashion-MNIST* (imágenes de prendas de vestir de $28 \times 28 \times 1$).
* **Backend**: Keras 3 operando sobre PyTorch (`os.environ["KERAS_BACKEND"] = "torch"`).
* **Arquitectura CNN**:
  1. `Conv2D` (32 filtros, $3\times3$, ReLU) + `MaxPooling2D` ($2\times2$) + `Dropout(0.25)`
  2. `Conv2D` (64 filtros, $3\times3$, ReLU) + `MaxPooling2D` ($2\times2$) + `Dropout(0.25)`
  3. `Flatten` + `Dense` (64 neuronas, ReLU) + `Dropout(0.50)`
  4. `Dense` (10 clases, Softmax)
* **Entrenamiento & Diagnóstico**: Optimizador `Adam`, función de pérdida `sparse_categorical_crossentropy`, reporte de clasificación completo (`precision`, `recall`, `f1-score`) y galería gráfica con predicciones coloreadas según acierto/fallo.

---

### 3. Deep Reinforcement Learning (DRL)

Ubicación: `DRL/`

Estudio práctico de agentes que aprenden interactuando con un entorno estocástico mediante prueba y error bajo el framework de **Gymnasium** (*FrozenLake-v1*).

#### 📊 Conceptos Clave Implementados
* **Ecuación de Bellman**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
* **Estrategia $\epsilon$-Greedy**: Balance dinámico entre **explorar** nuevas trayectorias y **explotar** el conocimiento acumulado mediante decaimiento exponencial ($\epsilon \cdot \text{decay}$).
* **Reward Shaping**: Modificación deliberada de las recompensas para penalizar pasos innecesarios ($-1$), castigar caídas ($-100$) y premiar la meta ($+100$).

#### 🤖 Implementaciones:
1. **`guia1.py` (Q-Learning Tabular)**:
   * Tabla $Q$ de dimensiones $|\mathcal{S}| \times |\mathcal{A}|$ ($16 \times 4$).
   * Actualización iterativa en 1,000 episodios.
2. **`guia2.py` (Deep Q-Learning / Aproximación con Redes)**:
   * Uso de `MLPRegressor` con capas `(32, 32)` entrenado progresivamente mediante `partial_fit`.
   * Normalización continua del estado discreto en coordenadas $[x, y]$ bidimensionales normalizadas.
   * Manejo de inicio en frío (*cold-start*).

---

## 🛠️ Stack Tecnológico

| Dominio | Herramientas y Librerías |
| :--- | :--- |
| **Lenguaje Base** | Python 3.10+ |
| **Machine Learning Clásico** | `scikit-learn`, `scipy` |
| **Deep Learning** | `keras` (v3), `torch` (PyTorch backend) |
| **Reinforcement Learning** | `gymnasium` |
| **Procesamiento de Datos** | `numpy`, `pandas` |
| **Visualización & Gráficos** | `matplotlib`, `pillow`, `PyQt6` |

---

## 🚀 Instalación y Requisitos

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/ML-Aprendizaje.git
cd ML-Aprendizaje
```

### 2. Crear y activar entorno virtual
```bash
# En Linux / macOS:
python3 -m venv env
source env/bin/activate

# En Windows:
python -m venv env
env\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

> **Nota:** Para ejecutar los scripts con Keras y PyTorch o renderizado visual de Gymnasium, asegúrate de contar con los paquetes correspondientes instalados en tu entorno (`torch`, `gymnasium`).

---

## 💻 Guía de Ejecución

A continuación se muestran ejemplos directos para probar cada módulo:

```bash
# 1. Regresión Lineal en Premier League
python "Linear Regressión/Premier league/main.py"

# 2. Clasificación de dígitos con Red Neuronal MLP
python "Redes neuronales/clasificar_digitos.py"

# 3. Autoencoder (compresión y reconstrucción de imágenes)
python "Redes neuronales/autoencoder_digits_etiquetado.py"

# 4. Pipeline CNN con Keras y PyTorch (Fashion-MNIST)
python "Redes neuronales/CNN/codigo_completo.py"

# 5. Aprendizaje por refuerzo con Q-Learning
python "DRL/guia1.py"

# 6. Aprendizaje por refuerzo con Red Neuronal (DQN)
python "DRL/guia2.py"
```

---

## 👥 Autores

Desarrollado y mantenido como espacio de investigación y aprendizaje en Machine Learning por:

* **Alexis Salazar** ([@alerrsi](https://github.com/alerrsi))
* **Víctor Rubilar**
* **Maximiliano Gonzalez**

---

<div align="center">
  <sub>⭐ Si este repositorio te resulta útil o de interés didáctico, ¡no olvides dejarle una estrella!</sub>
</div>
