# НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
import os
os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam  # Use tf.keras optimizer

# Вывод информации о доступных устройствах
physical_devices = tf.config.list_physical_devices('GPU')
# Проверка доступных устройств
print("Доступные устройства:", tf.config.experimental.list_physical_devices())

if not physical_devices:
    print("No GPU devices available.")
else:
    print("GPU devices:")
    for device in physical_devices:
        print(f"- {device.name} ({device.device_type})")

# Создаем среду LunarLander
env = gym.make("LunarLander-v2", render_mode="human")

# Определяем параметры DQN
num_episodes = 1000
gamma = 0.99  # discount factor
learning_rate = 0.001

# Создаем нейронную сеть (здесь упрощенная)
model = Sequential()
model.add(Dense(64, input_dim=env.observation_space.shape[0], activation="relu"))
model.add(Dense(env.action_space.n, activation="linear"))
model.compile(loss="mse", optimizer=Adam(learning_rate))

# Обучение
for episode in range(num_episodes):
    # Начальное состояние
    observation = env.reset()
    done = False
    total_reward = 0
    epsilon = 0.9

    while not done:
        # Выбираем действие (epsilon-greedy exploration)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            if isinstance(observation, tuple):  # Check if it's a tuple
                observation = np.array(observation[0])  # Convert to NumPy array

            q_values = model.predict(observation[np.newaxis, :])
            action = np.argmax(q_values)

        # Выполняем действие
        next_observation, reward, done, truncated, info = env.step(action)

        # Обновляем Q-values
        target = reward
        if not done:
            target += gamma * np.max(model.predict(next_observation[np.newaxis, :]))

        # Extract the list of floats
        observation_values = observation[0]
        if observation_values.shape != ():
            test = np.array(observation_values)
            test_1 = test[np.newaxis, :]
            # Handle the case where observation_values has only one element
            q_values = model.predict(test_1)

            q_values[0][action] = target
            model.fit(np.array(observation_values)[np.newaxis, :], q_values, epochs=1, verbose=0)

            # Переходим к следующему состоянию
            observation = next_observation
            total_reward += reward
        else:
            pass

    # Уменьшаем epsilon (exploration rate)
    epsilon *= 0.99

    # Выводим результаты
    print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()