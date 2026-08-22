import numpy as np

matriz_A = np.array([[1, 2], 
                     [3, 4]])
matriz_B = np.array([[5, 6], 
                     [7, 8]])

# 1. Multiplicación de matrices (NO es elemento por elemento, sigue las reglas del álgebra lineal)
multiplicacion = matriz_A @ matriz_B

print(multiplicacion)
transpuesta_A = matriz_A.T
transpuesta_B = matriz_B.T


multiplicacion_t = transpuesta_A @ transpuesta_B

print(multiplicacion_t)


vector_suma = np.array([10, 20])
# NumPy automáticamente suma 10 a la primera columna y 20 a la segunda
resultado_broadcasting = transpuesta_A + vector_suma
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
print(resultado_broadcasting)