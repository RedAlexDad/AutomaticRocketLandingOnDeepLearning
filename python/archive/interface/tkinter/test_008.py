import numpy as np
import tkinter as tk


class RocketLanderApp:
    def __init__(self, master, terrain_size, roughness):
        self.master = master
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.terrain = self.generate_terrain()

        # Создаем график рельефа
        self.plot_terrain()

        # Определяем начальные координаты ракеты
        rocket_width = 20  # Ширина ракеты (размер по оси X)
        rocket_height = 20  # Высота ракеты (размер по оси Y)
        rocket_start_x = 250 - rocket_width / 2  # Начальная позиция ракеты по оси X
        rocket_start_y = rocket_height  # Начальная позиция ракеты по оси Y (с учетом начала координат в верхнем левом углу холста)
        self.rocket = self.canvas.create_polygon(rocket_start_x, rocket_start_y,
                                                 rocket_start_x - rocket_width / 2, rocket_start_y + rocket_height,
                                                 rocket_start_x + rocket_width / 2, rocket_start_y + rocket_height,
                                                 fill="red")

        # Создаем ракету
        # self.rocket = self.canvas.create_polygon(250, 30, 240, 50, 260, 50, fill="red")

        self.canvas.bind("<KeyPress>", self.move_rocket)
        self.canvas.focus_set()

    def generate_terrain(self):
        terrain = np.zeros(self.terrain_size)
        terrain[0] = np.random.rand()
        terrain[-1] = np.random.rand()
        self.divide(terrain, 0, len(terrain) - 1, self.roughness)
        return terrain

    def divide(self, terrain, start, end, roughness):
        if end - start < 2:
            return
        middle = (start + end) // 2
        terrain[middle] = (terrain[start] + terrain[end]) / 2 + np.random.randn() * roughness
        self.divide(terrain, start, middle, roughness / 2)
        self.divide(terrain, middle, end, roughness / 2)

    def move_rocket(self, event):
        key = event.keysym
        x, y = self.canvas.coords(self.rocket)[0], 500 - self.canvas.coords(self.rocket)[1]  # Инвертируем координаты Y
        print(f'''position: x: {x}, y: {y}''')
        if key == "Left":
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right":
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up":
            self.canvas.move(self.rocket, 0, -5)
        elif key == "Down":
            self.canvas.move(self.rocket, 0, 5)  # Изменено на положительное значение для движения вниз

        # Проверка столкновения с поверхностью
        if y >= self.terrain[int(x) // 5] * 500:
            self.check_landing(y, self.terrain[int(x) // 5])

    def check_landing(self, rocket_y, terrain_height):
        landing_threshold = 5  # Пороговое значение для успешного приземления
        print('rocket_y:', rocket_y, 'terrain_height:', terrain_height, 'result:',
              abs(rocket_y - terrain_height) <= landing_threshold)
        if abs(rocket_y - terrain_height) <= landing_threshold:
            print("Приземление успешно!")
        elif rocket_y < terrain_height:
            print("Крах!")
        else:
            # print('Летим вниз!')
            pass

    def plot_terrain(self):
        x_scale = 500 / self.terrain_size
        y_scale = 500
        for i in range(1, len(self.terrain)):
            x1 = (i - 1) * x_scale
            y1 = 500 - self.terrain[i - 1] * y_scale  # Изменено на 500 минус высота
            x2 = i * x_scale
            y2 = 500 - self.terrain[i] * y_scale  # Изменено на 500 минус высота
            self.canvas.create_line(x1, y1, x2, y2, fill="black")

        # Ось координат X
        self.canvas.create_line(0, 490, 500, 490, fill="red", arrow=tk.LAST)
        for i in range(0, 500, 50):
            self.canvas.create_text(i, 490, anchor=tk.NW, text=str(i))

        # Ось координат Y
        self.canvas.create_line(2, 0, 2, 500, fill="blue", arrow=tk.FIRST)
        for i in range(0, 500, 50):
            self.canvas.create_text(5, 500 - i, anchor=tk.NW, text=str(i))


def main():
    root = tk.Tk()
    app = RocketLanderApp(root, 100, 0.7)
    root.mainloop()


if __name__ == "__main__":
    main()
