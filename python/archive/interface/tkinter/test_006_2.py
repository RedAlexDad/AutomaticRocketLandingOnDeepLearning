import numpy as np
import tkinter as tk


class RocketLanderApp:
    def __init__(self, master, terrain_size, roughness, width=1000, height=500):
        self.master = master
        self.canvas = tk.Canvas(master, width=width, height=height)
        self.canvas.pack()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.width = width
        self.height = height
        self.terrain = self.generate_terrain()

        # Создаем ракету
        self.rocket = self.canvas.create_polygon(250, self.height - 30, 240, self.height - 10, 260, self.height - 10,
                                                 fill="red")

        self.canvas.bind("<KeyPress>", self.move_rocket)
        self.canvas.focus_set()

        # Создаем график рельефа
        self.plot_terrain()

    def generate_terrain(self, max_height=0.5, min_height=0.5):
        terrain = np.zeros(self.terrain_size)

        terrain[0] = np.random.rand() * max_height  # Ограничение высоты
        terrain[-1] = np.random.rand() * min_height  # Ограничение высоты

        self.divide(terrain, 0, len(terrain) - 1, self.roughness)

        return terrain

    def divide(self, terrain, start, end, roughness, low=0, high=0.5):
        if end - start < 2:
            return
        middle = (start + end) // 2

        # Генерация случайной высоты в диапазоне от low до high
        terrain[middle] = (terrain[start] + terrain[end]) / 2 + np.random.uniform(low, high) * roughness

        # Ограничение высоты
        terrain[middle] = min(max(terrain[middle], low), high)

        self.divide(terrain, start, middle, roughness / 2)
        self.divide(terrain, middle, end, roughness / 2)

    def move_rocket(self, event):
        key = event.keysym
        x, y = self.canvas.coords(self.rocket)[0], self.canvas.coords(self.rocket)[1]

        # Получаем размеры ракеты
        rocket_width = 20  # Пример
        rocket_height = 20  # Пример

        # Проверяем, не выходит ли ракета за пределы холста
        if key == "Left" and x > 0:  # Проверяем, что x > 0, чтобы ракета не выходила за левый край
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right" and x < self.width - rocket_width:  # Проверяем, что x < ширина холста - ширина ракеты
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up" and y > 0:  # Проверяем, что y > 0, чтобы ракета не выходила за верхний край
            self.canvas.move(self.rocket, 0, -5)
        elif key == "Down" and y < self.height - rocket_height:  # Проверяем, что y < высота холста - высота ракеты
            self.canvas.move(self.rocket, 0, 5)

        print(f'''position: x: {x}, y: {y}''')

        # Проверка столкновения с поверхностью
        terrain_index = int(x * 100 / self.width)  # Преобразуем в проценты и умножаем на 100
        if terrain_index >= 0 and terrain_index < len(self.terrain):
            terrain_height = round(int(self.terrain[terrain_index]) * self.height, 3)
            if y <= terrain_height:
                self.check_landing(y, self.terrain[terrain_index])

    def check_landing(self, rocket_y, terrain_height):
        if rocket_y >= terrain_height * self.width:
            print("Приземление успешно!")
        else:  # Авария
            print("Крах!")

    def plot_terrain(self):
        x_scale = int(self.width / self.terrain_size)
        y_scale = int(self.width)

        for i in range(1, len(self.terrain)):
            x1 = (i - 1) * x_scale
            y1 = int(self.terrain[i - 1] * y_scale)

            x2 = i * x_scale
            y2 = int(self.terrain[i] * y_scale)

            self.canvas.create_line(x1, y1, x2, y2, fill="black")


def main():
    root = tk.Tk()
    app = RocketLanderApp(root, 10, 0.1, width=1200, height=600)
    root.mainloop()


if __name__ == "__main__":
    main()
