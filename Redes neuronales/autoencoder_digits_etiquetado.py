import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# CASO: RED NEURONAL + APRENDIZAJE SIN ETIQUETAS EXTERNAS
# ============================================================
# Problema: aprender una representación de imágenes de dígitos
#          que permita reconstruirlas.
#
# Tipo de aprendizaje:
#   NO SUPERVISADO, en el sentido utilizado en esta unidad:
#   no usamos las etiquetas 0, 1, 2, ..., 9 para entrenar.
#
# NOTA:
# En terminología moderna, un autoencoder también puede describirse
# como aprendizaje AUTO-SUPERVISADO, porque la propia entrada X
# se utiliza como objetivo de reconstrucción.
#
# Modelo:
#   AUTOENCODER construido con una red neuronal.
#
# IDEA CENTRAL:
#
#   Imagen original
#        ↓
#   CODIFICADOR
#        ↓
#   Código latente
#        ↓
#   DECODIFICADOR
#        ↓
#   Imagen reconstruida
#
# Luego comparamos:
#
#   Imagen original  <->  Imagen reconstruida
#
# para calcular el error de reconstrucción.
# ============================================================


# 1. OBTENER LOS DATOS
# load_digits() contiene imágenes pequeñas de dígitos manuscritos.
# Cada imagen tiene 8 x 8 píxeles.
datos = load_digits()

# X contiene las imágenes.
# Cada imagen 8x8 viene representada como un vector de 64 valores.
X = datos.data

# datos.target contiene las etiquetas 0, 1, 2, ..., 9.
#
# IMPORTANTE:
# En este ejercicio NO las utilizaremos para entrenar.
# Al autoencoder no le interesa saber si la imagen es un 3, 7 o 9.
#
# Su tarea es aprender a reconstruir la imagen.
y = datos.target


# 2. NORMALIZAR LOS VALORES DE LOS PÍXELES
# En digits, cada píxel tiene valores entre 0 y 16.
#
# Los convertimos al rango 0 a 1.
#
# Esto facilita el entrenamiento de la red neuronal.
X = X / 16.0


# 3. SEPARAR LOS DATOS EN ENTRENAMIENTO Y PRUEBA
# Aquí ocurre una diferencia importante respecto del MLP clasificador:
#
# Antes teníamos:
#   X_train, X_test, y_train, y_test
#
#       Recordemos que:
#           X_train: contiene los datos de entrada que se utilizan para entrenar el modelo. 
#                    En digits, son las imágenes representadas por sus 64 píxeles.
#           X_test:  contiene datos de entrada que el modelo no utilizó durante el entrenamiento. 
#                    Se usan después para comprobar cómo responde frente a ejemplos nuevos.
#           y_train: contiene las etiquetas correctas asociadas a cada elemento de X_train. 
#                    En digits, indica qué número representa cada imagen: 0, 1, 2, ..., 9.
#           y_test:  Contiene las etiquetas correctas asociadas a X_test. Se utilizan para comparar 
#                    las predicciones del modelo con las respuestas reales y evaluar su desempeño.
# 
# 
# Ahora solamente necesitamos:
#   X_train, X_test
#
# porque NO utilizaremos etiquetas para entrenar el autoencoder.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Total de imagenes usadas para entrenar:", len(X_train))
print("Total de imagenes usadas para probar:", len(X_test))


# 4. CREAR EL AUTOENCODER
#
# Utilizaremos MLPRegressor para construir el autoencoder. Aquí no queremos clasificar la 
# imagen en una de 10 categorías, como ocurría con MLPClassifier. Ahora queremos reconstruir 
# la imagen completa, por lo que la red debe producir nuevamente los 64 valores de la entrada 
# en la salida, uno por cada píxel de la imagen 8×8.
#
# Arquitectura:
#
#   64 entradas
#       ↓
#      32
#       ↓
#      16     <- CÓDIGO LATENTE / CUELLO DE BOTELLA
#       ↓
#      32
#       ↓
#   64 salidas
#
# La red "deshidrata" la imagen:
#
#   64 -> 32 -> 16
#
# y después intenta "rehidratarla":
#
#   16 -> 32 -> 64
#
# El valor 16 corresponde aquí al espacio latente:
# una representación más compacta de la imagen original.
autoencoder = MLPRegressor(
    hidden_layer_sizes=(32, 16, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)


# 5. ENTRENAR EL AUTOENCODER
#
# ESTA ES LA LÍNEA CLAVE DEL EJERCICIO:
#
#           X_train  ->  X_train
#
# La misma imagen se utiliza como:
#
#   ENTRADA   = X_train
#   OBJETIVO  = X_train
#
# No le decimos:
#   "esta imagen es un 7"
#
# Le decimos, conceptualmente:
#   "recibe esta imagen e intenta reconstruirla".
#
# Durante el entrenamiento ocurre:
#
#   1) La imagen entra a la red.
#   2) El CODIFICADOR reduce la información.
#   3) Se obtiene un CÓDIGO LATENTE.
#   4) El DECODIFICADOR intenta reconstruir la imagen.
#   5) Se compara la reconstrucción con la imagen original.
#   6) Se calcula el error de reconstrucción (loss).
#   7) BACKPROPAGATION propaga ese error hacia atrás.
#   8) Se ajustan los pesos.
#   9) El proceso se repite.
#
# El objetivo es reducir progresivamente el error de reconstrucción.
autoencoder.fit(X_train, X_train)


# 6. RECONSTRUIR IMÁGENES QUE LA RED NO USÓ PARA ENTRENAR
#
# X_test contiene imágenes nuevas para el modelo.
#
# El autoencoder recibe cada imagen y genera 64 valores de salida,
# que corresponden a su intento de reconstrucción.
X_reconstruido = autoencoder.predict(X_test)


# 7. MEDIR EL ERROR DE RECONSTRUCCIÓN
#
# En el MLP clasificador comparábamos:
#
#   predicción  <->  etiqueta correcta
#
# En el autoencoder comparamos:
#
#   imagen reconstruida  <->  imagen original
#
# mean_squared_error calcula qué tan diferentes son ambas.
#
# Mientras MÁS PEQUEÑO sea el error,
# más parecida es la reconstrucción a la entrada.
error = mean_squared_error(X_test, X_reconstruido)

print("Error medio de reconstrucción:", error)


# 8. MOSTRAR ALGUNAS IMÁGENES ORIGINALES Y RECONSTRUIDAS
#
# Primera fila:
#   imágenes originales
#
# Segunda fila:
#   imágenes reconstruidas por el autoencoder
#
# Así podemos evaluar visualmente qué información logró conservar
# el código latente.
cantidad = 20

plt.figure(figsize=(12, 4))

for i in range(cantidad):

    # Determinar en qué mitad estamos
    grupo = i // 10
    posicion = i % 10

    # ----------------------------
    # IMAGEN ORIGINAL
    # ----------------------------
    plt.subplot(4, 10, grupo * 20 + posicion + 1)
    imagen_original = X_test[i].reshape(8, 8)
    plt.imshow(imagen_original, cmap="Blues")
    # plt.title(f"Original\nDígito: {y_test[i]}")
    plt.title(f"Original")
    plt.axis("off")


    # ----------------------------
    # IMAGEN RECONSTRUIDA
    # ----------------------------
    plt.subplot(4, 10, grupo * 20 + posicion + 11)
    imagen_reconstruida = X_reconstruido[i].reshape(8, 8)

    plt.imshow(imagen_reconstruida, cmap="Blues")
    plt.title("Reconstruida")
    plt.axis("off")

    plt.title("Reconstruida")
    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# RESUMEN DEL CASO
# ============================================================
#
# RED NEURONAL:
#   Autoencoder
#
# ARQUITECTURA:
#   64 -> 32 -> 16 -> 32 -> 64
#
# CÓDIGO LATENTE:
#   16 valores
#
# TIPO DE APRENDIZAJE:
#   Sin etiquetas externas.
#   Para esta unidad: NO SUPERVISADO.
#   Más precisamente: AUTO-SUPERVISADO.
#
# OBJETIVO:
#   Reconstruir la imagen de entrada.
#
# ¿USA LAS ETIQUETAS 0-9?
#   NO.
#
# ¿QUÉ HACE EL CODIFICADOR?
#   "Deshidrata" o comprime la información de la entrada.
#
# ¿QUÉ ES EL CÓDIGO LATENTE?
#   Una representación numérica compacta aprendida por la red.
#
# ¿QUÉ HACE EL DECODIFICADOR?
#   Utiliza el código latente para intentar reconstruir la entrada.
#
# ¿DE DÓNDE SALE EL ERROR?
#   De comparar:
#
#       imagen reconstruida  <->  imagen original
#
# ¿QUÉ HACE BACKPROPAGATION?
#   Propaga hacia atrás la información del error para determinar
#   cómo deben modificarse los pesos de la red.
#
# ¿QUÉ HACE DESPUÉS EL OPTIMIZADOR?
#   Ajusta los pesos buscando disminuir el error.
#
#
# COMPARACIÓN CON EL EJERCICIO ANTERIOR
# ------------------------------------------------------------
#
# MLP CLASIFICADOR
#
#   Entrada:
#       imagen
#
#   Objetivo:
#       etiqueta 0-9
#
#   Aprende:
#       a CLASIFICAR
#
#   Entrenamiento:
#       modelo.fit(X_train, y_train)
#
#
# AUTOENCODER
#
#   Entrada:
#       imagen
#
#   Objetivo:
#       reconstruir la misma imagen
#
#   Aprende:
#       una REPRESENTACIÓN de la imagen
#
#   Entrenamiento:
#       autoencoder.fit(X_train, X_train)
#
#
# IDEA CLAVE:
#
#   El MLP que usamos aprende a decir QUÉ ES la imagen.
#
#   El autoencoder aprende a REPRESENTAR la imagen de manera
#   que pueda reconstruirla.
#
# ============================================================
