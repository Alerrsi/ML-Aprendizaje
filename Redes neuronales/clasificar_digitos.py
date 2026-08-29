import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay 

#Obtención de los datos
datos = ds.load_digits()
X = datos.data
Y = datos.target

print(f"X = {len(X)}")
print(f"Y = {len(Y)}")


for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(datos.images[i],cmap="gray")
    plt.title(datos.target[i])
    plt.axis("off")
#plt.show()

# Separar entrenamiento y pruebas

X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)

print(f"Total de imagenes usadas en entrenar: {len(X_train)}")
print(f"Total de imagenes usadas en probar : {len(X_test)}")
print(f"Total de etiquetas correctas que son de X_train: {len(Y_train)}")
print(f"Total de etiquetas correctas que son de X_test: {len(Y_test)}")

# Crear la red neuronal 

modelo = MLPClassifier(
    hidden_layer_sizes = (64,),
    #Ciclos de iteración (No se detiene)
    max_iter = 500, 
    random_state=42
)

# Entrenar el modelo

modelo.fit(X_train,Y_train)

# Predicción del modelo

predicciones = modelo.predict(X_test)

# Evaluar

print(f"Exactitud: {accuracy_score(Y_test,predicciones)}")

# Matriz de confusión

matriz = confusion_matrix(Y_test,predicciones)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=datos.target_names
)

disp.plot()
plt.xlabel("Etiquetas Predichas")
plt.ylabel("Etiquetas Reales")
plt.show()

# Buscar los casos donde el modelo se equivoca

errores = []

for i in range(len(Y_test)):
    if predicciones[i] != Y_test[i]:
        errores.append(i)

print(f"Catidad de errores: {len(errores)}")

# Mostrar visualmente los errores

for posicion, indice in enumerate(errores[:10]):
    plt.subplot(2,5, posicion + 1)
    imagen = X_test[indice].reshape(8,8)
    plt.imshow(imagen,cmap="Blues")
    plt.title(
        f"Real: {Y_test[indice]}\n"
        f"Predijo: {predicciones[indice]}"
    )
    plt.axis("off")
plt.show()

# Realizar arbol de desiciones en casa

