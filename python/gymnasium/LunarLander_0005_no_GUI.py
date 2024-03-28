import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class LunarLanderDQNAgent:
    def __init__(self, env_name="LunarLander-v2", num_episodes=1000, gamma=0.99, learning_rate=0.001,
                 CUDA_VISIBLE_DEVICES="0"):
        # НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
        os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']
        # Установите переменные окружения для использования процессора (при желании)
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

        # Окружающая среда и гиперпараметры для обучения
        self.env = gym.make(env_name, render_mode="human")
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Модель нейронной сети
        self.model = self.build_model()

    @staticmethod
    def check_gpu_tensorflow():
        # Вывод информации о доступных устройствах
        physical_devices = tf.config.list_physical_devices('GPU')
        # Проверка доступных устройств
        print("Доступные устройства:", tf.config.experimental.list_physical_devices())
        if physical_devices:
            print("GPU доступен для TensorFlow.")
            for device in physical_devices:
                print(f"- {device.name} ({device.device_type})")
        else:
            print("GPU не доступен для TensorFlow или не установлен.")

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.env.observation_space.shape[0], activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(self.learning_rate))
        return model

    def choose_action(self, observation, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            if isinstance(observation, tuple):
                observation = np.array(observation[0])
            q_values = self.model.predict(observation[np.newaxis, :], verbose=0)
            return np.argmax(q_values)

    def update_q_values(self, observation, action, reward, next_observation, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.model.predict(next_observation[np.newaxis, :], verbose=0))

        observation_values = observation[0]
        if observation_values.shape != ():
            test = np.array(observation_values)
            test_1 = test[np.newaxis, :]
            q_values = self.model.predict(test_1, verbose=0)
            q_values[0][action] = target
            self.model.fit(np.array(observation_values)[np.newaxis, :], q_values, epochs=1, verbose=0)

    def train(self):
        # Инициализация данных построения
        self.episode_rewards = []

        for episode in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            epsilon = 0.9

            while not done:
                action = self.choose_action(observation, epsilon)
                next_observation, reward, done, truncated, info = self.env.step(action)
                self.update_q_values(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward

            # Обновление данных графика
            self.episode_rewards.append(total_reward)

            epsilon *= 0.99
            print(f"Episode: {episode}, Total Reward: {total_reward}")

        print('Медианные результаты обучения:', np.median(self.episode_rewards))
        self.env.close()

    def plot_rewards(self):
        epochs = range(1, self.num_episodes + 1)
        plt.plot(epochs, self.episode_rewards, marker='o', linestyle='-')
        plt.title('Суммарная награда за каждую эпоху')
        plt.xlabel('Эпоха')
        plt.ylabel('Суммарная награда')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    agent = LunarLanderDQNAgent(num_episodes=5)
    agent.check_gpu_tensorflow()
    agent.train()
    agent.plot_rewards()