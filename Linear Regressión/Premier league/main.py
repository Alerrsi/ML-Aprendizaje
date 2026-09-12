import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# carga de datos
data = pd.read_csv("final_data.csv")

data.columns = data.columns.str.replace(" ", "_")


print(data.columns)

# variables predictoras (features)
features = data.drop(
    columns=[
        "player",
        "team",
        "name",
        "position",
        "height",
        "appearance",
        "current_value",
    ]
)


target = data["current_value"]

# Preparamos el  entrenamiento del modelo
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# creación del modelo
model = LinearRegression(fit_intercept=True)

# Entrenamos el modelo
model.fit(X_train, Y_train)

predicciones = model.predict(X_test)

print(f"Exactitud: {np.sqrt(mean_squared_error(Y_test, predicciones))}")
print(predicciones)

print("originales")
print(Y_test)


for i in range(1, 20):
    print(round(predicciones[i]), "           ", Y_test.iloc[i])
