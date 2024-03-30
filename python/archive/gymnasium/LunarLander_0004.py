import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import threading


class LunarLanderTrainer:
    def __init__(self, model, env, num_episodes):
        self.model = model
        self.env = env
        self.num_episodes = num_episodes

    def train(self):
        observations = []
        actions = []
        rewards = []

        # Обучение модели
        start_time = time.time()
        for episode in range(self.num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            while True:
                # Собираем данные для обучения
                observations.append(observation)
                action = self.env.action_space.sample()  # здесь вы можете использовать вашу модель для предсказания действия
                actions.append(action)

                # Выполняем действие в среде
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    # Если эпизод завершился, начинаем новый
                    observation = self.env.reset()
                    episode_reward = 0
                    break

            rewards.append(episode_reward)

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


# Функция для отображения окна среды
def render_environment(env):
    env.render()


# Функция для отображения графика метрики
def plot_metric(rewards):
    plt.plot(rewards)
    plt.xlabel('Эпизоды')
    plt.ylabel('Награда')
    plt.title('Изменение награды во время обучения')
    plt.show()


# Создание модели и среды
model = LinearRegression()
env = gym.make("LunarLander-v2",render_mode="human")

# Создание экземпляра класса LunarLanderTrainer и обучение модели
trainer = LunarLanderTrainer(model, env, num_episodes=100)
train_thread = threading.Thread(target=trainer.train)
train_thread.start()

# Отображение окна среды в основном потоке
while True:
    render_environment(env)
    time.sleep(0.01)  # Задержка для стабильного обновления окна среды

# Тестирование обученной модели
trainer.test()

# Закрытие окна среды
env.close()
