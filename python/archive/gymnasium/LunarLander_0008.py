import os
import threading

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class LunarLanderDQNAgent:
    def __init__(self, env_name="LunarLander-v2", num_episodes=1000, gamma=0.99, learning_rate=0.001,
                 CUDA_VISIBLE_DEVICES="0"):
        # Set environment variables for CPU usage (if desired)
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        # НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
        os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']

        # Окружающая среда и гиперпараметры для обучения
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
        # Create a separate thread for the GUI (Tkinter doesn't require a separate thread)
        gui_thread = threading.Thread(target=self.create_gui)
        gui_thread.start()

        # Initialize plot data
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

            # Выдает сигнал для обновления графика в потоке графического интерфейса пользователя
            self.update_plot(episode_rewards)

            epsilon *= 0.99
            print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.env.close()

    def update_plot(self, episode_rewards):
        # Clear the previous plot
        self.ax.clear()

        # Plot the new reward data
        self.ax.plot(episode_rewards)
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        # Force the plot to redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def create_gui(self):
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Lunar Lander DQN Agent")
        self.root.geometry("500x500")

        # Create matplotlib figure and axes
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Interactive mode for dynamic updates

        # Create a canvas to embed the plot in the GUI
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect the signal to the update_plot slot function
        self.update_plot(self.update_plot)

        self.root.mainloop()  # Start the Tkinter event loop


if __name__ == "__main__":
    # Create and train the agent
    agent = LunarLanderDQNAgent()
    agent.train()