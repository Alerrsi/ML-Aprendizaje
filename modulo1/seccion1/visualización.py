import numpy as np
import matplotlib.pyplot as plt

# 1. Configurar la figura que contendrá ambos gráficos
fig = plt.figure(figsize=(12, 6))

# ==========================================
# GRÁFICO 1: Plano 2D (2 Dimensiones)
# ==========================================
# Añadir un subplot (1 fila, 2 columnas, posición 1)
ax1 = fig.add_subplot(121)

# Definir vectores 2D: [x, y]
v1_2d = np.array([3, 4])
v2_2d = np.array([-4, 2])
origen_2d = np.array([0, 0]) # Los vectores nacen en el origen

# Dibujar las flechas (quiver)
# scale=1, scale_units='xy', angles='xy' es para que la escala de la flecha sea exacta a los ejes
ax1.quiver(*origen_2d, *v1_2d, color='red', scale=1, scale_units='xy', angles='xy', label='v1 (3, 4)')
ax1.quiver(*origen_2d, *v2_2d, color='blue', scale=1, scale_units='xy', angles='xy', label='v2 (-4, 2)')

# Configurar el plano 2D
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.axhline(0, color='black', linewidth=1) # Eje X
ax1.axvline(0, color='black', linewidth=1) # Eje Y
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_aspect('equal') # Cuadrícula perfectamente cuadrada
ax1.set_title('Plano 2D (Vectores [x, y])')
ax1.legend()


# ==========================================
# GRÁFICO 2: Espacio 3D (3 Dimensiones)
# ==========================================
# Añadir un subplot con proyección 3D (1 fila, 2 columnas, posición 2)
ax2 = fig.add_subplot(122, projection='3d')

# Definir vectores 3D: [x, y, z]
v1_3d = np.array([3, 2, 5])
v2_3d = np.array([-3, 4, 2])
origen_3d = np.array([0, 0, 0]) # Origen en 3D

# En 3D, quiver toma los puntos de inicio (x,y,z) y las direcciones (u,v,w)
ax2.quiver(*origen_3d, *v1_3d, color='red', arrow_length_ratio=0.1, label='v1 (3, 2, 5)')
ax2.quiver(*origen_3d, *v2_3d, color='blue', arrow_length_ratio=0.1, label='v2 (-3, 4, 2)')

# Configurar el espacio 3D
ax2.set_xlim([-6, 6])
ax2.set_ylim([-6, 6])
ax2.set_zlim([0, 6])
ax2.set_xlabel('Eje X')
ax2.set_ylabel('Eje Y')
ax2.set_zlabel('Eje Z')
ax2.set_title('Espacio 3D (Vectores [x, y, z])')
ax2.legend()

# Mostrar ambos gráficos en pantalla
plt.tight_layout()
plt.show()