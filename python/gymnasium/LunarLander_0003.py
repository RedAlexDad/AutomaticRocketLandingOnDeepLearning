import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time


class LunarLanderTrainer:
    def __init__(self, model, env, num_episodes):
        self.model = model
        self.env = env
        self.num_episodes = num_episodes

    def train(self):
        observations = []
        actions = []
        rewards = []

        # Инициализация графика
        plt.ion()  # Включаем интерактивный режим для отображения графика в реальном времени
        fig, ax = plt.subplots()
        rewards_line, = ax.plot([], [], '-o')
        ax.set_xlabel('Эпизоды')
        ax.set_ylabel('Награда')
        ax.set_title('Изменение награды во время обучения')
        fig.canvas.draw()
        plt.show()

        # Обучение и тестирование модели
        start_time = time.time()
        for episode in range(self.num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            while True:
                # Отображение окна среды
                self.env.render()

                # Собираем данные для обучения
                observations.append(observation)
                action = self.env.action_space.sample()  # здесь вы можете использовать вашу модель для предсказания действия
                actions.append(action)

                # Выполняем действие в среде
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                # Обновляем график награды
                rewards.append(episode_reward)
                rewards_line.set_xdata(range(len(rewards)))
                rewards_line.set_ydata(rewards)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()

                if terminated or truncated:
                    # Если эпизод завершился, начинаем новый
                    observation = self.env.reset()
                    episode_reward = 0
                    break

        # Преобразуем данные в numpy массивы
        X = np.array(observations)
        y = np.array(actions)

        # Обучаем модель на собранных данных
        self.model.fit(X, y)

        end_time = time.time()
        print(f"Обучение завершено за {end_time - start_time} секунд.")

    def test(self):
        observation = self.env.reset()
        while True:
            # Предсказываем действие с помощью обученной модели
            action = self.model.predict(np.array([observation]))[0]

            # Выполняем действие в среде
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation = self.env.reset()


# Создание модели и среды
model = LinearRegression()
env = gym.make("LunarLander-v2", render_mode="human")

# Создание экземпляра класса LunarLanderTrainer и обучение модели
trainer = LunarLanderTrainer(model, env, num_episodes=1000)
trainer.train()

# Тестирование обученной модели
trainer.test()

# Закрытие окна среды
env.close()
