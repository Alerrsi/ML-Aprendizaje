# Álgebra Lineal: Vectores y Matrices

Los vectores y las matrices son las estructuras matemáticas fundamentales detrás del **Machine Learning** y la **Ciencia de Datos**, utilizadas para representar datos, transformaciones espaciales y parámetros de modelos.

---

## 1. Vectores

Un vector puede entenderse desde múltiples perspectivas complementarias:

* **Perspectiva Geométrica:** Es un punto en un espacio $n$-dimensional (por ejemplo, el plano $2\text{D}$ o el espacio $3\text{D}$), o una flecha dirigida desde el origen $(0, 0)$ hacia dicho punto con una **magnitud** (longitud) y **dirección**.
* **Perspectiva Informática / Programación:** Es una lista o arreglo unidimensional ordenado de números (ej. `[x, y]`).

### Notaciones Principales

| Enfoque | Notación | Representación |
| :--- | :--- | :--- |
| **Física** | Con flecha superior | $\vec{v}$ |
| **Álgebra Lineal (Vector Columna)** | Matriz de $n \times 1$ | $\vec{v} = \begin{bmatrix} x \\ y \end{bmatrix}$ |
| **Álgebra Lineal (Vector Fila)** | Matriz transpuesta de $1 \times n$ | $\vec{v}^T = \begin{bmatrix} x & y \end{bmatrix}$ |
| **Programación (Python / NumPy)** | Array unidimensional | `v = np.array([x, y])` |

![Representación de vectores](images/image.png)

---

## 2. Operaciones Fundamentales con Vectores

Las dos operaciones elementales que definen el espacio vectorial son la **suma de vectores** y la **multiplicación por un escalar**.

### 2.1. Suma de Vectores

La suma de dos vectores de la misma dimensión se realiza **elemento por elemento** (*element-wise*).

#### Formulación Matemática
Dados dos vectores $\vec{u} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}$ y $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$:

$$\vec{u} + \vec{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

#### Ejemplo Numérico
$$\begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ -2 \end{bmatrix} = \begin{bmatrix} 2 + 5 \\ 3 + (-2) \end{bmatrix} = \begin{bmatrix} 7 \\ 1 \end{bmatrix}$$

> **Interpretación geométrica:** Visualmente corresponde a la regla del paralelogramo o método punta-cola (colocar el origen del segundo vector sobre el extremo final del primero).

---

### 2.2. Multiplicación por un Escalar

Un **escalar** es un número real ($c \in \mathbb{R}$). Al multiplicar un vector por un escalar, cada una de sus componentes se multiplica por dicho número, modificando la magnitud (tamaño) y/o el sentido del vector.

#### Formulación Matemática
$$c \cdot \vec{v} = c \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} c \cdot a \\ c \cdot b \end{bmatrix}$$

#### Comportamiento según el valor de $c$:

1. **Amplificación ($|c| > 1$):** El vector se alarga manteniendo su orientación.
   $$2 \cdot \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} 2a \\ 2b \end{bmatrix}$$

2. **Reducción ($0 < |c| < 1$):** El vector se contrae proporcionalmente.
   $$\frac{1}{2} \cdot \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} \frac{1}{2}a \\ \frac{1}{2}b \end{bmatrix}$$

3. **Inversión y Escalado ($c < 0$):** El vector invierte su sentido ($180^\circ$) y modifica su magnitud según el valor absoluto de $c$.
   $$-2.3 \cdot \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} -2.3a \\ -2.3b \end{bmatrix}$$

![Multiplicación por un escalar](images/image2.png)

---

## 3. Introducción a Matrices

Una **matriz** es una cuadrícula bidimensional de números de tamaño $m \times n$ ($m$ filas y $n$ columnas).

$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$$

### Operaciones Clave con Matrices

* **Transposición ($A^T$):** Intercambia filas por columnas.
  $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$

* **Multiplicación Matricial ($A \cdot B$ o `@` en NumPy):** No es elemento a elemento; se obtiene calculando el producto punto entre las filas de la primera matriz y las columnas de la segunda.

* **Broadcasting:** Técnica en NumPy que permite realizar operaciones aritméticas entre arreglos/matrices de diferentes formas compatibles (por ejemplo, sumar un vector a cada fila de una matriz).
