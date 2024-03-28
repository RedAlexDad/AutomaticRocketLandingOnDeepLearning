import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LunarLanderDQNAgent:
    def __init__(self, env_name="LunarLander-v2", num_episodes=10, gamma=0.99, learning_rate=0.001):
        # Окружающая среда и гиперпараметры для обучения
        # Set environment variables for CPU usage (if desired)
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        # НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
        os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']

        self.env = gym.make(env_name, render_mode="human")
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Модель нейронной сети
        self.model = self.build_model()

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
            q_values = self.model.predict(observation[np.newaxis, :])
            return np.argmax(q_values)

    def update_q_values(self, observation, action, reward, next_observation, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.model.predict(next_observation[np.newaxis, :]))

        observation_values = observation[0]
        if observation_values.shape != ():
            test = np.array(observation_values)
            test_1 = test[np.newaxis, :]
            q_values = self.model.predict(test_1)
            q_values[0][action] = target
            self.model.fit(np.array(observation_values)[np.newaxis, :], q_values, epochs=1, verbose=0)

    def train(self):
        # Инициализация данных построения
        episode_rewards = []

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
            episode_rewards.append(total_reward)

            epsilon *= 0.99
            print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.env.close()

        # Построение графического интерфейса после обучения
        self.create_gui(episode_rewards)

    def create_gui(self, episode_rewards):
        root = tk.Tk()
        root.title("Lunar Lander DQN Agent")
        root.geometry("800x600")

        # Создание графика
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Отрисовка графика
        x_values = range(len(episode_rewards))
        y_values = episode_rewards
        canvas.create_line(*zip(*((x, 600 - y * 10) for x, y in zip(x_values, y_values))), fill="blue", width=2)

        root.mainloop()


if __name__ == "__main__":
    # Создание и обучение агента
    agent = LunarLanderDQNAgent()
    agent.train()
