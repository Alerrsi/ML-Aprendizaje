# apuntes de guia 1 generada de DRL
#
#
#


# 3 piezas del aprendizaje por refuerzo
# Estado → Estado del entorno actual
# Acción → lo que el agente puede hacer en ese estado
# Función de recompenssa la regla que castiga o premia
# al agente segun el resultado de su acción

# Q es que tan conveniente estima el modelo segun (estado, acción)
# Deep-@-network es la red neuronl que aproxima los valores en vez de
# guardarlos en una tabla
#


# Explorar vs Explotar
# Explorar → es realizar una acción al azar sin usar lo aprendido
# para descubrir posiblidades
# Explotar → Elegir una acción con el mayor valor Q conocido
# hasta el momento, se usa lo ya aprendido
# Epsilon → probabilidad de explorar en vez de explotar
# 1 100% de explorar
# 0 100% de explotar
#
# Epsilon-greedy
# tecnica donde se empieza con un Epsilon cercano a 1 para que el modelo explore
# y con el pasar de episodios baje para que confie en sus habilidades
#

# Recompensa
# Un egente no piensa solo en la recompensa inmediata mas bien
# tiene tambien en cuenta la recompensa a futuro
# para calcular la recompensa de cada paso se usa recompensa + Y * valorQ futura
# gamma = y | valores entre 0 y
# mas cercano a 1 valora mas recompensa a futuro
# mas cercano a cero es cortoplacista
#
#


# Implementación de ejemplo en codigo
#
import gymnasium as gym
import numpy as np

#              cuadricula 4x4   sin resbalones
env = gym.make("FrozenLake-v1", is_slippery=False)


# caracteristicas del entorno
n_acciones = env.action_space.n
n_estados = env.observation_space.n

# tabla Q
Q = np.zeros((n_estados, n_acciones))
# hiperparametros de entrenamiento
# son valores externos a el modelo como configuraciones
# los parametros comunes son aquellos que el modeloa aprende entrenando

alpha = 0.8  # taza de aprendizaje
gamma = 0.95  # importancia de Q futuro
epsilon = 1.0  # probabilidad de explorar
epsilon_min = 0.01  # piso minimo de exploracion
epsilon_decay = 0.995  # que tan rapido baja epsilon
episodios = 2000  # cuantas partidas completas se juegan
max_pasos = 10  # limite de pasos por partida

# recompensas por episodio

recompensas_por_episodio = []
recompensa = 0

for episodio in range(1000):
    estado, _ = env.reset()  # extraemos solo el estado
    recompensa_total = 0

    for paso in range(max_pasos):  # cada episodio da 100 pasos
        random = np.random.rand()
        # print(f"numero al azar: {random} | epsilon: {epsilon}")
        if random < epsilon:  # numero al azar
            accion = env.action_space.sample()  # explorar
            # print("Explorar")

        else:
            # print("Explotar")
            accion = np.argmax(Q[estado])  # explotar
            # print("aaa", np.argmax(Q[estado]))  # explotar

        nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)
        # Moldeado de recompensas ajustado
        if not terminado:
            recompensa = -1
        elif terminado and recompensa == 0:
            recompensa = -100
        elif terminado and recompensa == 1:
            recompensa = 100
        print(f"actual {Q[estado, accion]}")
        # print(env.step(accion))

        mejor_Q_siguiente = np.max(Q[nuevo_estado])
        Q[estado, accion] = Q[estado, accion] + alpha * (
            # Q objetivo                              Q estimado
            recompensa + gamma * mejor_Q_siguiente - Q[estado, accion]
        )
        print(
            f"tranformado{
                Q[estado, accion]
                + alpha
                * (
                    # Q objetivo                              Q estimado
                    recompensa + gamma * mejor_Q_siguiente - Q[estado, accion]
                )
            }"
        )

        print(f"Q objetivo {recompensa + gamma * mejor_Q_siguiente}")
        print("error", recompensa + gamma * mejor_Q_siguiente - Q[estado, accion])
        print("_________________________")
        # print(f"estado actual{estado}")
        # print(f"nuevo estado{nuevo_estado} | recompensa: {recompensa}")
        estado = nuevo_estado

        recompensa_total += recompensa

        if terminado or truncado:
            break

    #   print("-----------------")

    # print(f"recompensa total: {recompensa_total} | episodio: {episodio} ")
    # print("________________________________________________________")

    # si el epsilon baja del minimo se queda fijo en este
    # si min => 0.1 y epsilon => 0.09, como es mayor min se queda allí
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # agregamos la recompensa generada por cada episodio
    recompensas_por_episodio.append(recompensa_total)
