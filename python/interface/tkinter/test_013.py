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

        # Определяем начальные координаты ракеты
        self.rocket_width = 20  # Ширина ракеты (размер по оси X)
        self.rocket_height = 20  # Высота ракеты (размер по оси Y)
        rocket_start_x = self.width / 2 - self.rocket_width / 2  # Начальная позиция ракеты по оси X
        rocket_start_y = self.rocket_height  # Начальная позиция ракеты по оси Y (с учетом начала координат в верхнем левом углу холста)

        self.rocket = self.canvas.create_polygon(
            rocket_start_x, rocket_start_y,
            rocket_start_x - self.rocket_width / 2, rocket_start_y + self.rocket_height,
            rocket_start_x + self.rocket_width / 2, rocket_start_y + self.rocket_height,
            fill="red"
        )

        self.canvas.bind("<KeyPress>", self.move_rocket)
        self.canvas.focus_set()

        # Создаем график рельефа
        self.plot_terrain()

    def generate_terrain(self, max_height=0.3, min_height=0.1):
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
        terrain[middle] = ((terrain[start] + terrain[end]) /
                           2 + np.random.uniform(low, high) * roughness)

        # Ограничение высоты
        terrain[middle] = min(max(terrain[middle], low), high)

        self.divide(terrain, start, middle, roughness / 2)
        self.divide(terrain, middle, end, roughness / 2)

    def move_rocket(self, event):
        key = event.keysym

        rocket_coords = self.canvas.coords(self.rocket)
        # Средняя координата X и округляем до трех знаков после запятой
        x = round((rocket_coords[0] + rocket_coords[2] + rocket_coords[4]) / 3, 3)
        # Средняя координата Y с учетом начала координат в левом нижнем углу и округляем до трех знаков после запятой
        y = round(self.height - (rocket_coords[1] + rocket_coords[3] + rocket_coords[5]) / 3, 3)

        # Проверяем, не выходит ли ракета за пределы холста
        if key == "Left" and x > 0:  # Проверяем, что x > 0, чтобы ракета не выходила за левый край
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right" and x < self.width - self.rocket_width:  # Проверяем, что x < ширина холста - ширина ракеты
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up" and y > 0:  # Проверяем, что y > 0, чтобы ракета не выходила за верхний край
            self.canvas.move(self.rocket, 0, -5)
        elif key == "Down" and y < self.height - self.rocket_height:  # Проверяем, что y < высота холста - высота ракеты
            self.canvas.move(self.rocket, 0, 5)

        # left_distance, right_distance, points = self.calculate_diagonal_distance(x, y)

        # Проверка столкновения с поверхностью
        terrain_index = int(x * 100 / self.width)  # Преобразуем в проценты и умножаем на 100

        # Проверка столкновения с поверхностью и округление до трех знаков после запятой
        terrain_height = round(self.terrain[min(int(x) // 5, len(self.terrain) - 1)] * self.width, 3)

        print(f'position: x: {x}, y: {y}; terrain_height: {terrain_index}, terrain_height: {terrain_height}, {self.terrain[terrain_index]}')
        # print(f'position: x: {x}, y: {y}; terrain_height: {terrain_index}, left_distance: {left_distance}, right_distance: {right_distance}')

        if terrain_index >= 0 and terrain_index < len(self.terrain):
            terrain_height = round(int(self.terrain[terrain_index]) * self.height, 3)
            if y <= terrain_height:
                self.check_landing(y, self.terrain[terrain_index])

    def check_landing(self, rocket_y, terrain_height):
        landing_threshold = 5  # Пороговое значение для успешного приземления
        if abs(rocket_y - terrain_height) <= landing_threshold:
            print("Приземление успешно!")
        elif rocket_y < terrain_height:
            print("Крах!")
        else:
            # print('Летим вниз!')
            pass

    def calculate_diagonal_distance(self, rocket_x, rocket_y):
        """
        Вычисляет расстояния по диагонали от ракеты до поверхности рельефа слева и справа,
        а также возвращает массив точек для рисования на canvas.

        Args:
            rocket_x: координата X ракеты.
            rocket_y: координата Y ракеты.

        Returns:
            Кортеж из трех элементов:
                - left_distance: расстояние по диагонали слева.
                - right_distance: расстояние по диагонали справа.
                - points: массив точек для рисования на canvas.
        """

        # Определяем индексы точек рельефа слева и справа от ракеты
        left_index = int(rocket_x) // 5
        right_index = left_index + 1

        # Проверяем, не вышли ли мы за пределы массива terrain
        if right_index >= len(self.terrain):
            right_index = len(self.terrain) - 1

        # Вычисляем высоты рельефа слева и справа от ракеты
        left_height = self.terrain[left_index] * self.height
        right_height = self.terrain[right_index] * self.height

        # Вычисляем горизонтальное расстояние до точек рельефа
        left_distance = abs(rocket_x - left_index * 5)
        right_distance = abs(rocket_x - right_index * 5)

        # Вычисляем вертикальные расстояния до точек рельефа
        left_vertical_distance = abs(rocket_y - left_height)
        right_vertical_distance = abs(rocket_y - right_height)

        # Вычисляем расстояния по диагонали до точек рельефа
        left_diagonal_distance = np.sqrt(left_distance ** 2 + left_vertical_distance ** 2)
        right_diagonal_distance = np.sqrt(right_distance ** 2 + right_vertical_distance ** 2)

        # Создаем массив точек для рисования на canvas
        points = [
            rocket_x, rocket_y,  # Координаты ракеты
            left_index * 5, left_height,  # Координаты левой точки рельефа
            right_index * 5, right_height  # Координаты правой точки рельефа
        ]

        # Возвращаем результаты
        return left_diagonal_distance, right_diagonal_distance, points


    def plot_terrain(self):
        x_scale = int(self.width / self.terrain_size)
        y_scale = int(self.width)

        for i in range(1, len(self.terrain)):
            x1 = (i - 1) * x_scale
            y1 = self.height - int(self.terrain[i - 1] * y_scale)

            x2 = i * x_scale
            y2 = self.height - int(self.terrain[i] * y_scale)

            self.canvas.create_line(x1, y1, x2, y2, fill="black")

        # Ось координат X
        self.canvas.create_line(0, self.height - 10, self.width, self.height - 10, fill="red", arrow=tk.LAST)
        for i in range(0, self.width, 50):
            self.canvas.create_text(i, self.height - 10, anchor=tk.NW, text=str(i))

        # Ось координат Y
        self.canvas.create_line(2, 0, 2, self.height, fill="blue", arrow=tk.FIRST)
        for i in range(0, self.height, 50):
            self.canvas.create_text(5, self.height - i, anchor=tk.NW, text=str(i))


def main():
    root = tk.Tk()
    app = RocketLanderApp(root, 100, 0.1, width=1200, height=600)
    root.mainloop()


if __name__ == "__main__":
    main()
