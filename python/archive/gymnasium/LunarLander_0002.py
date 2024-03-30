import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Создаем среду
env = gym.make("LunarLander-v2", render_mode="human")

# Инициализируем модель машинного обучения
model = LinearRegression()

# Массивы для хранения данных об обучении
observations = []
actions = []
rewards = []  # Массив для хранения наград за каждый эпизод

# Обучение модели
observation = env.reset()
episode_reward = 0

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
for episode in range(100):  # Задаем количество эпизодов для обучения
    while True:
        # Отображение окна среды
        env.render()

        # Собираем данные для обучения
        observations.append(observation)
        action = env.action_space.sample()  # здесь вы можете использовать вашу модель для предсказания действия
        actions.append(action)

        # Выполняем действие в среде
        observation, reward, terminated, truncated, info = env.step(action)
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
            observation = env.reset()
            episode_reward = 0
            break

# Преобразуем данные в numpy массивы
X = np.array(observations)
y = np.array(actions)

# Обучаем модель на собранных данных
model.fit(X, y)

# Теперь модель обучена и может использоваться для предсказания действий в среде

# Теперь давайте протестируем обученную модель в среде
observation = env.reset()
while True:
    # Предсказываем действие с помощью обученной модели
    action = model.predict(np.array([observation]))[0]

    # Выполняем действие в среде
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation = env.reset()

# Закрытие окна среды
env.close()
