import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tabulate import tabulate

from tensorflow.keras.layers import Dense, Dropout
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

        # Данные
        self.dataset = {
            'epochs': [],
            'rewards': [],
            'actions': [],
            'observations': [],
            'next_observations': [],
            'dones': []
        }

    def update_dataset(self, epoch, total_reward, actions, observations, next_observations, dones):
        self.dataset['epochs'] = epoch
        self.dataset['rewards'] = total_reward
        self.dataset['actions'].extend(actions)
        self.dataset['observations'].extend(observations)
        self.dataset['next_observations'].extend(next_observations)
        self.dataset['dones'].extend(dones)

    def print_dataset(self):
        headers = ['Epoch', 'Reward', 'Actions', 'Observations', 'Next Observations', 'Done']
        data = []
        for i in range(len(self.dataset['epochs'])):
            epoch = self.dataset['epochs'][i]
            reward = self.dataset['rewards'][i]
            actions = self.dataset['actions'][i]
            observations = self.dataset['observations'][i]
            next_observations = self.dataset['next_observations'][i]
            done = self.dataset['dones'][i]
            data.append([epoch, reward, actions, observations, next_observations, done])
        print(tabulate(data, headers=headers, tablefmt='grid'))

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
        model.add(Dense(256, input_dim=self.env.observation_space.shape[0], activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))  # Добавим слой Dropout для регуляризации
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

        for epoch in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            epsilon = 0.9
            epochs_array = []
            total_rewards_array = []
            actions = []
            observations = []
            next_observations = []
            dones = []

            while not done:
                action = self.choose_action(observation, epsilon)
                next_observation, reward, done, truncated, info = self.env.step(action)
                actions.append(action)
                observations.append(observation)
                next_observations.append(next_observation)
                dones.append(done)
                self.update_q_values(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward
                epochs_array.append(epoch)
                total_rewards_array.append(total_reward)

            # Обновление данных графика
            self.episode_rewards.append(total_reward)
            # Обновление датасета
            self.update_dataset(epochs_array, total_rewards_array, actions, observations, next_observations, dones)

            epsilon *= 0.99
            print(f"Эпоха: {epoch}, Суммарная награда: {total_reward}")

        self.env.close()

    def plot_rewards(self):
        epochs = range(1, self.num_episodes + 1)
        plt.plot(epochs, self.episode_rewards, marker='o', linestyle='-')
        plt.title('Суммарная награда за каждую эпоху')
        plt.xlabel('Эпоха')
        plt.ylabel('Суммарная награда')
        plt.grid(True)
        plt.show()

    def plot_rewards_realtime(self):
        plt.figure()
        plt.title('Суммарная награда за каждую эпоху')
        plt.xlabel('Эпоха')
        plt.ylabel('Суммарная награда')
        plt.grid(True)
        plt.ion()  # Включаем интерактивный режим

        epochs = []
        epochs_array = []
        self.episode_rewards = [] # Сохраняем суммарную награду
        rewards = []
        moving_average = []  # Для хранения скользящего среднего
        window_size = 5  # Размер окна для скользящего среднего

        line, = plt.plot(epochs, rewards, marker='o', linestyle='-', label='Суммарная награда')  # Инициализируем график
        moving_avg_line, = plt.plot(epochs, moving_average, color='red', linestyle='--',
                                    label='Скользящее среднее')  # Линия для скользящего среднего
        plt.legend()  # Добавляем легенду
        plt.show()

        for epoch in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            epsilon = 0.9
            actions = []
            observations = []
            next_observations = []
            dones = []

            while not done:
                action = self.choose_action(observation, epsilon)
                next_observation, reward, done, truncated, info = self.env.step(action)
                actions.append(action)
                observations.append(observation)
                next_observations.append(next_observation)
                dones.append(done)
                self.update_q_values(observation, action, reward, next_observation, done)
                observation = next_observation
                total_reward += reward
                rewards.append(reward)
                epochs_array.append(epoch)

            # Обновляем данные для графика
            epochs.append(epoch + 1)
            self.episode_rewards.append(total_reward)

            # Добавляем текущее значение в скользящее среднее
            if len(rewards) >= window_size:
                moving_average.append(np.mean(rewards[-window_size:]))
            else:
                moving_average.append(np.mean(rewards))

            # Обновление датасета
            self.update_dataset(epochs_array, rewards, actions, observations, next_observations, dones)

            # Обновляем график в реальном времени
            line.set_xdata(epochs)
            line.set_ydata(self.episode_rewards)
            moving_avg_line.set_xdata(epochs)
            moving_avg_line.set_ydata(moving_average)
            plt.gca().relim()  # Обновляем пределы осей для корректного отображения
            plt.gca().autoscale_view(True, True, True)  # Автоматически масштабируем оси
            plt.draw()
            plt.pause(0.001)  # Приостанавливаем выполнение программы на короткое время для обновления графика

            if (epoch + 1 == self.num_episodes):
                break

        plt.ioff()  # Выключаем интерактивный режим после завершения
        plt.show()

    def save_dataset_to_excel(self, filename, type_='csv'):
        df = pd.DataFrame({
            'Epochs': self.dataset['epochs'],
            'Rewards': self.dataset['rewards'],
            'Actions': self.dataset['actions'],
            'Observations': self.dataset['observations'],
            'Next Observations': self.dataset['next_observations'],
            'Dones': self.dataset['dones']
        })

        if type_ == 'csv':
            df.to_csv(filename + '.csv', index=False)
        else:
            df.to_excel(filename + '.xlsx', index=False)


if __name__ == "__main__":
    agent = LunarLanderDQNAgent(num_episodes=1000)
    agent.check_gpu_tensorflow()
    agent.plot_rewards_realtime()
    # agent.print_dataset()
    agent.save_dataset_to_excel(filename='dataset', type_='xlsx')
