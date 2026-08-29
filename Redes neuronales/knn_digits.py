import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# carga de datos
datos = ds.load_digits()


# variables predictoras
X = datos.data
# variable objetivo
Y = datos.target

modelo = KNeighborsClassifier(
    n_neighbors=5,
)

# separación de datos de entrenamiento y de prueba
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Total de imagenes usadas en entrenar: {len(X_train)}")
print(f"Total de imagenes usadas en probar : {len(X_test)}")
print(f"Total de etiquetas correctas que son de X_train: {len(Y_train)}")
print(f"Total de etiquetas correctas que son de X_test: {len(Y_test)}")


modelo.fit(X_train, Y_train)


predicciones = modelo.predict(X_test)


print(f"Exactitud: {accuracy_score(Y_test, predicciones)}")

# Matriz de confusión

matriz = confusion_matrix(Y_test, predicciones)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz, display_labels=datos.target_names
)

disp.plot()
plt.xlabel("Etiquetas Predichas")
plt.ylabel("Etiquetas Reales")
plt.show()


# cantidad de errores
errores = [error for error in Y_test]
