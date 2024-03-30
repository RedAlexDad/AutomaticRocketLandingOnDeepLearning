import math
import sys
import random

import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QGridLayout,
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygonF, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QPoint


class Rocket(QWidget):
    def __init__(self, terrain_generator, parent=None):
        super().__init__(parent)

        self.terrain_generator = terrain_generator

        # Начальные координаты ракеты
        self.x = self.terrain_generator.width() // 2
        # Коррекция координаты Y
        self.y = int(self.terrain_generator.get_terrain_height(self.x))

        # Состояние ракеты (приземление или крах)
        self.landing_state = "Flying"  # Изначально ракета в полете

    def paintEvent(self, event):
        painter = QPainter(self)

        # Отрисовка ракеты в виде треугольника
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)

        # Определяем точки треугольника
        triangle = QPolygonF()
        triangle.append(QPointF(self.x, self.y))
        triangle.append(QPointF(self.x - 5, self.y + 10))
        triangle.append(QPointF(self.x + 5, self.y + 10))

        # Рисуем треугольник
        painter.drawPolygon(triangle)

        # Проверяем состояние приземления
        if self.landing_state == "Landing":
            painter.drawText(self.width() - 100, 20, "Landing...")
        elif self.landing_state == "Crash":
            painter.drawText(self.width() - 100, 20, "Crash!")
        elif self.landing_state == "Flying":
            painter.drawText(self.width() - 100, 20, "Flying...")

    def move_rocket(self, delta_x, delta_y):
        # Перемещение ракеты по оси X и Y, при условии, что она не выходит за пределы экрана
        new_x = self.x + delta_x
        new_y = self.y + delta_y

        # Проверяем, достигла ли ракета поверхности
        if new_y <= self.terrain_generator.get_terrain_height(new_x):
            # Проверяем, находится ли ракета в пределах безопасной зоны для приземления
            if abs(self.terrain_generator.get_terrain_height(new_x) - new_y) <= 5:
                self.landing_state = "Landing"
            elif abs(self.terrain_generator.get_terrain_height(new_x) - new_y) > 5:
                self.landing_state = "Crash"
            else:
                self.landing_state = "Flying"

        if 0 <= new_x <= self.terrain_generator.width() and 0 <= new_y <= self.terrain_generator.height():
            self.x = new_x
            self.y = new_y
            self.update()
class TerrainGenerator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Настройки рельефа

        # Увеличьте количество очков для более ровного рельефа
        self.num_points = 50
        self.terrain_height = 100

        # Генерация функции рельефа
        self.generate_terrain_function()

        # Генерация рельефа при запуске программы
        self.generate_terrain()

    def generate_terrain_function(self):
        # Example: Sinusoidal function with some noise
        def terrain_function(x):
            # Sinusoidal component
            sinusoidal = 50 * math.sin(x / 50)

            # Random noise component
            noise = random.randint(-10, 10)

            # Gaussian hill component
            hill = 30 * math.exp(-(x - 300) ** 2 / (2 * 100 ** 2))

            # Combination of components
            return sinusoidal + noise + hill

        self.terrain_function = terrain_function

    def generate_terrain(self):
        # Generate terrain points based on the function
        qpoint_list = []
        for i in range(self.num_points):
            x = i * self.width() / (self.num_points - 1)
            # Коррекция вычисления координаты Y
            y = self.get_terrain_height(x)
            qpoint_list.append(QPoint(x, y))

        self.terrain_polygon = QPolygonF(qpoint_list)

    def get_terrain_height(self, x):
        # Получить координату y (высоту) для заданной координаты x
        return self.terrain_function(x)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Отрисовка рельефа
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        # Создайте QPainterPath из полигона
        path = QPainterPath()
        path.addPolygon(self.terrain_polygon)

        # Заполнить контур (область рельефа)
        brush = QBrush(Qt.darkGreen)
        painter.fillPath(path, brush)


class CoordinateGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Настройки сетки
        self.grid_size = 20
        self.grid_color = Qt.gray

        # Настройки метки координат ракеты
        self.rocket_coordinates = QPoint(0, 0)
        self.rocket_label = QLabel("Rocket: (0, 0)")

    def paintEvent(self, event):
        painter = QPainter(self)

        # Отрисовка сетки
        pen = QPen(self.grid_color, 1, Qt.DashLine)
        painter.setPen(pen)
        for x in range(0, self.width(), self.grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), self.grid_size):
            painter.drawLine(0, y, self.width(), y)

    def resizeEvent(self, event):
        # Обновление размеров и позиции метки ракеты при изменении размера виджета
        self.rocket_coordinates = QPoint(self.width() - 100, self.height() - 20)

    def set_rocket_coordinates(self, x, y):
        # Установка координат ракеты и обновление отображения
        self.rocket_coordinates = QPoint(x, y)
        self.update()


class MainWindow(QWidget):
    def __init__(self, width=400, height=400):
        super().__init__()
        # Установка ширины и высоты окна
        self.setFixedSize(width, height)
        # Создание виджета генератора рельефа
        self.terrain_generator = TerrainGenerator()
        # Создание виджета координатной сетки
        self.coordinate_grid = CoordinateGrid()
        # Создание метки координат
        self.coordinates_label = QLabel("X: 0, Y: 0")
        # Создание ракеты
        self.rocket = Rocket(self.terrain_generator)
        # Макет
        layout = QGridLayout()
        layout.addWidget(self.coordinate_grid, 0, 0)
        layout.addWidget(self.terrain_generator, 0, 0)
        layout.addWidget(self.rocket, 0, 0)  # Добавляем ракету на виджет
        layout.addWidget(self.coordinates_label, 0, 1, Qt.AlignCenter)
        self.setLayout(layout)

    def paintEvent(self, event):
        # Перерисовка элементов при перерисовке окна
        self.rocket.update()  # Обновляем ракету
        super().paintEvent(event)

    def keyPressEvent(self, event):
        # Обработка нажатий клавиш для управления ракетой
        if event.key() == Qt.Key_Left:
            self.rocket.move_rocket(-10, 0)  # Перемещение ракеты влево
        elif event.key() == Qt.Key_Right:
            self.rocket.move_rocket(10, 0)  # Перемещение ракеты вправо
        elif event.key() == Qt.Key_Up:
            self.rocket.move_rocket(0, -10)  # Перемещение ракеты вверх
        elif event.key() == Qt.Key_Down:
            self.rocket.move_rocket(0, 10)  # Перемещение ракеты вниз

        # Обновление метки координат ракеты
        self.update_rocket_coordinates_label()

    def update_rocket_coordinates_label(self):
        # Обновление метки координат ракеты
        self.coordinates_label.setText(f"Rocket: ({self.rocket.x}, {self.rocket.y})")
        print('Rocket coordinates: x = ', self.rocket.x, 'y = ', self.rocket.y)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Указание ширины и высоты окна при создании экземпляра MainWindow
    # window = MainWindow(width=1920, height=1080)
    window = MainWindow(width=int(1920 / 2), height=int(1080 / 2))

    window.show()
    sys.exit(app.exec_())