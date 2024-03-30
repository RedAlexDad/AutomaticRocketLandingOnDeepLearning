import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression

# Создаем среду
env = gym.make("LunarLander-v2", render_mode="human")

# Инициализируем модель машинного обучения
model = LinearRegression()

# Массивы для хранения данных об обучении
observations = []
actions = []

# Обучение модели
observation, info = env.reset(seed=42)
for _ in range(1000):
    # Собираем данные для обучения
    observations.append(observation)
    action = env.action_space.sample()  # здесь вы можете использовать вашу модель для предсказания действия
    actions.append(action)

    # Выполняем действие в среде
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        # Если эпизод завершился, начинаем новый
        observation, info = env.reset()

# Преобразуем данные в numpy массивы
X = np.array(observations)
y = np.array(actions)

# Обучаем модель на собранных данных
model.fit(X, y)

# Теперь модель обучена и может использоваться для предсказания действий в среде

# Теперь давайте протестируем обученную модель в среде
observation, info = env.reset(seed=42)
for _ in range(100):
    # Предсказываем действие с помощью обученной модели
    action = model.predict(np.array([observation]))[0]

    # Выполняем действие в среде
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
