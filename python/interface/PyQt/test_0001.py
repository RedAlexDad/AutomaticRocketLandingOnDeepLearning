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
from PyQt5.QtCore import Qt, QPointF

class TerrainGenerator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Настройки рельефа
        self.num_points = 10  # Increase number of points for smoother terrain
        self.terrain_height = 100

        # Генерация функции рельефа
        self.generate_terrain_function()

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

    def get_terrain_height(self, x):
        # Get the y-coordinate (height) for a given x-coordinate
        return self.terrain_function(x)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Отрисовка рельефа
        pen = QPen(Qt.black, 2)
        painter.setPen(pen)

        # Generate terrain points based on the function
        qpoint_list = []
        for i in range(self.num_points):
            x = i * self.width() / (self.num_points - 1)
            y = self.height() - self.get_terrain_height(x)
            qpoint_list.append(QPointF(x, y))

        polygon = QPolygonF(qpoint_list)

        # Create a QPainterPath from the polygon
        path = QPainterPath()
        path.addPolygon(polygon)

        # Fill the path (terrain area)
        brush = QBrush(Qt.darkGreen)
        painter.fillPath(path, brush)

class CoordinateGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Настройки сетки
        self.grid_size = 20
        self.grid_color = Qt.gray

        # Настройки центральной точки
        self.center_x = self.width() // 2
        self.center_y = self.height() // 2
        self.center_color = Qt.red

    def paintEvent(self, event):
        painter = QPainter(self)

        # Отрисовка сетки
        pen = QPen(self.grid_color, 1, Qt.DashLine)
        painter.setPen(pen)
        for x in range(0, self.width(), self.grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), self.grid_size):
            painter.drawLine(0, y, self.width(), y)

        # Отрисовка центральной точки
        brush = QBrush(self.center_color)
        painter.setBrush(brush)
        painter.drawEllipse(self.center_x - 5, self.center_y - 5, 10, 10)

    def resizeEvent(self, event):
        # Обновление центральной точки при изменении размера
        self.center_x = self.width() // 2
        self.center_y = self.height() // 2


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

        # Макет
        layout = QGridLayout()
        layout.addWidget(self.coordinate_grid, 0, 0)
        layout.addWidget(self.terrain_generator, 0, 0)
        layout.addWidget(self.coordinates_label, 0, 1, Qt.AlignCenter)
        self.setLayout(layout)

        # Подключение обработчика событий мыши
        self.coordinate_grid.mouseMoveEvent = self.update_coordinates

    def update_coordinates(self, event):
        # Обновление метки координат
        x = event.x() - self.coordinate_grid.center_x
        y = self.coordinate_grid.center_y - event.y()
        self.coordinates_label.setText(f"X: {x}, Y: {y}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(width=int(1920 / 2), height=int(1080 / 2))

    window.show()
    sys.exit(app.exec_())