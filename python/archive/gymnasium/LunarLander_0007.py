import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject, Qt, pyqtSignal, QTimer
from PyQt5.QtChart import (
    QChart,
    QChartView,
    QLineSeries, QValueAxis
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LunarLanderDQNAgent(QObject):
    # Определите сигнал обновления графика внутри класса
    plot_update_signal = pyqtSignal(list)

    def __init__(self, env_name="LunarLander-v2", num_episodes=1000, gamma=0.99, learning_rate=0.001,
                 CUDA_VISIBLE_DEVICES="0"):
        super().__init__()
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

        # Store a reference to the chart object
        self.chart = None
        # Connect the signal to the update_plot slot function
        self.plot_update_signal.connect(self.update_plot)

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

            # Выдает сигнал для обновления графика
            self.plot_update_signal.emit(episode_rewards)

            epsilon *= 0.99
            print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.env.close()

    def update_plot(self, episode_rewards):
        # Get the QLineSeries object from the chart
        series = self.chart.series()[0]

        # Clear the previous data
        series.clear()

        # Append the new reward data
        for i, reward in enumerate(episode_rewards):
            series.append(i, reward)

    def create_gui(self):
        # Create PyQt application and window
        app = QApplication([])
        window = QWidget()
        window.setWindowTitle("Lunar Lander DQN Agent")
        window.resize(500, 500)

        # Create chart and series
        chart = QChart()
        # Assign the chart object
        self.chart = chart
        chart.setTitle("Episode Rewards")
        series = QLineSeries()
        chart.addSeries(series)

        # Create and customize axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Epoch")
        axis_x.setRange(0, self.num_episodes)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Total Reward")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # Create chart view and layout
        chart_view = QChartView(chart)
        layout = QVBoxLayout()
        layout.addWidget(chart_view)
        window.setLayout(layout)

        window.show()

        # Запускаем обучение после создания GUI
        self.train()

        app.exec_()


if __name__ == "__main__":
    # Create and train the agent
    agent = LunarLanderDQNAgent()
    agent.create_gui()
