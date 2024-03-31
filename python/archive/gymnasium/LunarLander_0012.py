import os
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


class DuelingDoubleDeepQNetwork(tf.keras.Model):
    def __init__(self, num_actions: int, fc1: int, fc2: int):
        super(DuelingDoubleDeepQNetwork, self).__init__()
        self.dense1 = Dense(fc1, activation='relu')
        self.dense2 = Dense(fc2, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        avg_A = tf.math.reduce_mean(A, axis=1, keepdims=True)
        Q = (V + (A - avg_A))

        return Q, A


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
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.update_rate = 120
        self.step_counter = 0

        self.buffer = ReplayBuffer(100000, input_dim)

        self.q_net = DuelingDoubleDeepQNetwork(num_actions, 128, 128)
        self.q_target_net = DuelingDoubleDeepQNetwork(num_actions, 128, 128)
        self.q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.q_target_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def store_tuple(self, state: np.ndarray, action: int, reward: float, new_state: np.ndarray, done: bool):
        state = np.array(state[0])
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def policy(self, observation: np.array):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation.reshape(1, -1)])
            _, actions = self.q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0][0]

        return action

    def train_neural_network(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_rate == 0:
            self.q_target_net.set_weights(self.q_net.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted, _ = self.q_net(state_batch)
        q_next, _ = self.q_target_net(new_state_batch)
        q_target = q_predicted.numpy()
        _, actions = self.q_net(new_state_batch)
        max_actions = tf.math.argmax(actions, axis=1)

        for idx in range(done_batch.shape[0]):
            q_target[idx, action_batch[idx]] = \
                (reward_batch[idx] + self.discount_factor * q_next[idx, max_actions[idx]] * (1 - int(done_batch[idx])))

        self.q_net.train_on_batch(state_batch, q_target)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.step_counter += 1

    def train_model(self, env, num_episodes: int, avg_score_min: float = 200, metric_avg_score: str = 'mean',
                    score_min: float = 250,
                    graph: bool = False, table: bool = False):
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        f = 0
        txt = open("saved_networks_0002.txt", "w")

        for i in range(num_episodes):
            done = False
            score = 0.0
            state = env.reset()[0]
            while not done:
                action = self.policy(state)
                new_state, reward, done, info, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, new_state, done)
                state = new_state
                self.train_neural_network()

            scores.append(score)
            obj.append(goal)
            episodes.append(i)
            # Метрика средней оценки или медианы если metric_avg_score =='median'
            avg_score = np.mean(scores[-100:]) if metric_avg_score == 'mean' else np.median(scores[-100:])
            avg_scores.append(avg_score)
            print(
                f"Эпизод {i}/{num_episodes}, Оценка: {score} (Эпсилон: {self.epsilon}), {'Средняя оценка' if metric_avg_score == 'mean' else 'Медианная оценка'}: {avg_score}")

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

    def test_model(self, env, num_episodes: int, file_name: str, file_type: str = 'tf', avg_score_min: float = 200,
                   metric_avg_score: str = 'mean', score_min: float = 250, graph: bool = False, table: bool = False):
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file_name)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.q_net.load_weights(file_name)

        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0

        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0

            while not done:
                env.render()
                action = self.policy(state)
                new_state, reward, done, _ = env.step(action)
                episode_score += reward
                state = new_state

            score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

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
    lunar_lander.train_model(env=env, num_episodes=1000, metric_avg_score='median', graph=True, table=True)
    # lunar_lander.save_dataframe("dataset", format='excel')
    # lunar_lander.test_model(env=env, num_episodes=10, file_type='h5', file_name='trained_model_tf_hotz.zip', metric_avg_score='median', graph=True, table=True)
    # lunar_lander.save_dataframe("dataset", format='excel')
