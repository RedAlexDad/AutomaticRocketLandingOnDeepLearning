import os
import sys

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tabulate import tabulate
from replay_buffer import ReplayBuffer

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from stable_baselines3 import A2C


class LunarLander:
    def __init__(self, lr: float, discount_factor: float, num_actions: int, epsilon: float, batch_size: int,
                 input_dim: list[int], CUDA_VISIBLE_DEVICES: str = '0'):

        # НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
        os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']
        # Установите переменные окружения для использования процессора (при желании)
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.update_rate = 120
        self.step_counter = 0

        self.buffer = ReplayBuffer(100000, input_dim)

        # Создание модели нейронной сети
        self.model = self.build_model(input_dim, num_actions)

    def store_tuple(self, state: np.ndarray, action: int, reward: float, new_state: np.ndarray, done: bool):
        state = np.array(state[0])
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def build_model(self, input_dim, num_actions):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=input_dim),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(256, activation='sigmoid'),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def choose_action(self, state):
        # Реализация выбора действия с учетом epsilon-greedy стратегии
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = self.model.predict(state[np.newaxis, :], verbose=0)
            return np.argmax(q_values)
    def update_model(self):
        # Обновление модели с использованием мини-батчей из буфера воспроизведения
        batch = self.buffer.sample_buffer(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        q_values_next = self.model.predict(next_states, verbose=0)
        max_q_values_next = np.max(q_values_next, axis=1)

        q_values = self.model.predict(states, verbose=0)
        q_values[np.arange(len(actions)), actions] = rewards + (1 - dones) * self.discount_factor * max_q_values_next

        self.model.fit(states, q_values, verbose=0)

    def train_model(self, env, num_episodes: int, save_path: str = 'trained_model_tf') -> None:
        os.system('cls' if os.name == 'nt' else 'clear')  # Очищаем консоль
        score = 0.0

        for _ in range(num_episodes):
            done = False
            obs = env.reset()
            total_reward = 0.0

            print('Эпизод:', num_episodes)
            while not done:
                action = self.choose_action(obs)
                next_obs, reward, done, info, _ = env.step(action)
                self.store_tuple(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                # Перезаписываем предыдущий вывод
                print('')
                print('\033[F', end='')  # Сдвигаем курсор на одну строку вверх
                print('total_reward:', round(total_reward, 2), 'action:', action, 'reward:', round(reward, 2), 'done:', done)
                # print('obs:', obs, 'next_obs:', next_obs, 'info:', info)

                if len(self.buffer) > self.batch_size:
                    self.update_model()

            score += total_reward # Обновляем счет

        # Сохранение модели после завершения обучения
        self.model.save(save_path)

    def test_model(self, env, num_episodes: int, save_path: str) -> None:
        if not os.path.exists(f'{save_path}') and save_path != '':
            self.model.load(save_path)
            env = self.model.get_env()

        for episode in range(num_episodes):
            total_reward = 0
            obs = env.reset()
            done = False

            while not done:
                action = self.choose_action(obs)
                obs, reward, done, info, _ = env.step(action)
                total_reward += reward
                env.render()

            print(f"Эпизод {episode + 1}: награда = {total_reward}")

        env.close()

    def train_model_neural_network(self, env, num_episodes: int, avg_score_min: float = 200, metric_avg_score: str = 'mean',
                    score_min: float = 250,
                    graph: bool = False, table: bool = False):
        os.system('cls' if os.name == 'nt' else 'clear')  # Очищаем консоль

        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        f = 0
        txt = open("saved_networks_0002.txt", "w")

        for i in range(num_episodes):
            done = False
            score = 0.0
            obs = env.reset()
            while not done:
                action = self.choose_action(obs)
                next_obs, reward, done, info, _ = env.step(action)
                self.store_tuple(obs, action, reward, next_obs, done)
                score += reward
                obs = next_obs

                # Перезаписываем предыдущий вывод
                print('\033[F', end='')  # Сдвигаем курсор на одну строку вверх
                print('score:', round(score, 2), 'action:', action, 'reward:', round(reward, 2), 'done:', done)
                # print('obs:', obs, 'next_obs:', next_obs, 'info:', info)

                if len(self.buffer) > self.batch_size:
                    self.update_model()

            scores.append(score)
            obj.append(goal)
            episodes.append(i)
            # Метрика средней оценки или медианы если metric_avg_score =='median'
            avg_score = np.mean(scores[-100:]) if metric_avg_score == 'mean' else np.median(scores[-100:])
            avg_scores.append(avg_score)
            print(
                f"Эпизод {i}/{num_episodes}, Оценка: {score} (Эпсилон: {self.epsilon}), {'Средняя оценка' if metric_avg_score == 'mean' else 'Медианная оценка'}: {avg_score}")
            txt.write(
                f"Эпизод {i}/{num_episodes}, Оценка: {score} (Эпсилон: {self.epsilon}), {'Средняя оценка' if metric_avg_score == 'mean' else 'Медианная оценка'}: {avg_score}\n")

            # Сохранение модели и ее весов
            if avg_score >= avg_score_min and score >= score_min:
                lunar_lander.save_model_and_weights(f"saved_networks/d3qn_model{f}",
                                                    f"saved_networks/d3qn_model{f}/net_weights{f}.h5")

        txt.close()
        # Сохранение результатов
        self.df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

        if graph:
            plt.plot('x', 'Score', data=self.df, marker='', color='blue', linewidth=2, label='Оценка')
            plt.plot('x', 'Average Score', data=self.df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='Средняя оценка' if metric_avg_score == 'mean' else 'Медианная оценка')
            plt.plot('x', 'Solved Requirement', data=self.df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Требование к решению')
            plt.legend()
            plt.savefig('LunarLander_Train_0002.png')

        if table:
            print(tabulate(self.df, headers='keys', tablefmt='grid'))

    def test_model_neural_network(self, env, num_episodes: int, file_name: str, file_type: str = 'tf', avg_score_min: float = 200,
                   metric_avg_score: str = 'mean', score_min: float = 250, graph: bool = False, table: bool = False):
        if not os.path.exists(f'{file_name}') and file_name != '':
            self.model.load(file_name)

        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0

        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_score = 0.0

            while not done:
                env.render()
                action = self.choose_action(obs)
                new_obs, reward, done, info, _ = env.step(action)
                episode_score += reward
                obs = new_obs

            score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(episode)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

            print(f"Эпизод {episode + 1}: награда = {episode_score}")

        env.close()

    def save_model_and_weights(self, model_filename, weights_filename):
        """
        Метод для сохранения модели и ее весов.

        Параметры:
            - model_filename: str, имя файла для сохранения модели.
            - weights_filename: str, имя файла для сохранения весов модели.
        """
        self.q_net.save(model_filename)
        self.q_net.save_weights(weights_filename)
        print(f"Модель сохранена как {model_filename}, а ее веса как {weights_filename}")

    def save_dataframe(self, filename, format='csv'):
        """
        Метод для сохранения DataFrame в формате CSV или Excel.

        Параметры:
            - df: pandas.DataFrame, DataFrame, который нужно сохранить.
            - filename: str, имя файла для сохранения.
            - format: str, формат файла ('csv' или 'excel'). По умолчанию 'csv'.
        """
        if format == 'csv':
            self.df.to_csv(filename + '.csv', index=False)
            print(f"DataFrame сохранен в формате CSV как {filename}.csv")
        elif format == 'excel':
            self.df.to_excel(filename + '.xlsx', index=False)
            print(f"DataFrame сохранен в формате Excel как {filename}.xlsx")
        else:
            print("Неподдерживаемый формат. Допустимые значения: 'csv', 'excel'")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode='rgb_array')
    lunar_lander = LunarLander(lr=0.00075, discount_factor=0.99, num_actions=4, epsilon=1.0, batch_size=128,
                               input_dim=[8], CUDA_VISIBLE_DEVICES="0")
    lunar_lander.train_model_neural_network(env=env, num_episodes=100, metric_avg_score='median', graph=True, table=True)
    # lunar_lander.train_model(env=env, num_episodes=50)

    env = gym.make("LunarLander-v2", render_mode='human')
    # lunar_lander.test_model(env=env, num_episodes=50, save_path='trained_model_tf')

    lunar_lander.save_dataframe("dataset_0003", format='excel')
    lunar_lander.test_model_neural_network(env=env, num_episodes=10, file_type='h5', file_name='trained_model_tf_hotz.zip', metric_avg_score='median', graph=True, table=True)
    # lunar_lander.save_dataframe("dataset", format='excel')
