import gymnasium as gym
import numpy as np
from sklearn.neural_network import MLPRegressor

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")


estado_inicial, _ = env.reset()


# cerrr entorno
env.close()


# traduccion de estados
#
def convertir_estado(estado):
    """
    Tomamos el estado de 0-16 y transformamos en coordenadas
    fila = division entera
    columna = resto
    9 // 4 = 2
    9 % 4 = 1
    estado traducido = 9 → [2, 1]
    """
    fila = estado // 4
    columna = estado % 4

    return np.array([[fila / 3.0, columna / 3.0]])


print(convertir_estado(9))

"""
El modelo de DQN debe ser parcialmente entrenado antes de ser usado
puesto que no puede arrancar sin experiencias previas
lo que se conoce como 'Cold start'
"""

modelo_Q = MLPRegressor(
    hidden_layer_sizes=(32, 32), learning_rate_init=0.001, random_state=42
)

estados_init = []
q_init = []

for i in range(16):
    estado_convertido = convertir_estado(i)[0]
    estados_init.append(estado_convertido)
    q_init.append([0.0, 0.0, 0.0, 0.0])

print(estados_init)
print(q_init)


# convertir a formaro numpy
#
estados_init = np.array(estados_init)
q_init = np.array(q_init)

# entrenar el modelo
modelo_Q.partial_fit(estados_init, q_init)
estado_prueba = convertir_estado(4)
q_valores = modelo_Q.predict(estado_prueba)[0]


print(f"Valores Q predichos para el estado 0: {q_valores}")
print("================================\n")


gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
episodios = 1000
max_pasos = 100


recompensas_episodio = []

print("__INIANDO EN ENTRENAMIENTO__")


estado, _ = env.reset()
recompensa_total = 0

for i in range(episodios):
    estado_red = convertir_estado(estado)
    q_valores_actuales = modelo_Q.predict(estado_red)[0]

    if np.random.rand() < epsilon:
        accion = env.action_space.sample()  # Explorar: acción aleatoria
    else:
        accion = np.argmax(q_valores_actuales)

    # 3. Ejecutar la acción en el entorno Gymnasium (¡Solo una vez!)
    nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)

    # 4. Sistema de Moldeado de Recompensas (Reward Shaping)
    # Adaptamos las recompensas predeterminadas de FrozenLake para guiar al agente
    if not terminado:
        recompensa_mod = (
            -1
        )  # Castigo leve por cada paso (incentiva buscar el camino corto)
    elif terminado and recompensa == 0:
        recompensa_mod = -100  # Castigo fuerte por caer en un agujero
    elif terminado and recompensa == 1:
        recompensa_mod = 100  # Gran premio por llegar a la meta

        # 5. Calcular el Q-objetivo (Ecuación de Bellman)
    if terminado or truncado:
        q_objetivo = recompensa_mod
    else:
        # Consultamos los valores Q del nuevo estado para encontrar el mejor futuro
        q_valores_siguientes = modelo_Q.predict(convertir_estado(nuevo_estado))[0]
        mejor_Q_siguiente = np.max(q_valores_siguientes)
        q_objetivo = recompensa_mod + gamma * mejor_Q_siguiente

        # 6. Preparar el vector Q para entrenar
        # Copiamos las predicciones actuales y actualizamos SOLO la acción que se ejecutó
    q_valores_entrenamiento = q_valores_actuales.copy()
    q_valores_entrenamiento[accion] = q_objetivo

    # 7. Actualizar la red neuronal mediante partial_fit
    # La red ajusta sus pesos internos corrigiendo la estimación hacia el q_objetivo
    modelo_Q.partial_fit(estado_red, [q_valores_entrenamiento])

    # Avanzar al siguiente estado
    estado = nuevo_estado
    recompensa_total += recompensa_mod

    if terminado or truncado:
        break

        # Reducir epsilon gradualmente al finalizar cada episodio
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    recompensas_por_episodio.append(recompensa_total)

    # Mostrar progreso cada 100 episodios
    if (episodio + 1) % 100 == 0:
        print(
            f"Episodio {episodio + 1:4d} | Epsilon: {epsilon:.3f} | Recompensa Total: {recompensa_total}"
        )

    print("¡Entrenamiento finalizado con éxito!")
