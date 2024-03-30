import sys
import numpy as np
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem, \
    QGraphicsLineItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QPointF


class RocketLanderApp(QMainWindow):
    def __init__(self, terrain_size, roughness, width=1000, height=500):
        super().__init__()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.width = width
        self.height = height
        self.terrain = self.generate_terrain()
        self.rocket_width = 20
        self.rocket_height = 20
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Rocket Lander")
        self.setGeometry(100, 100, self.width + 50, self.height + 50)
        self.scene = QGraphicsScene(0, 0, self.width, self.height)
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.draw_terrain()
        self.draw_axes()
        self.rocket_item = self.draw_rocket()
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.keyPressEvent = self.move_rocket

    def generate_terrain(self, max_height=0.3, min_height=0.1):
        terrain = np.zeros(self.terrain_size)
        terrain[0] = np.random.rand() * max_height
        terrain[-1] = np.random.rand() * min_height
        self.divide(terrain, 0, len(terrain) - 1, self.roughness)
        return terrain

    def divide(self, terrain, start, end, roughness, low=0, high=0.2):
        if end - start < 2:
            return
        middle = (start + end) // 2
        terrain[middle] = ((terrain[start] + terrain[end]) / 2 + np.random.uniform(low, high) * roughness)
        terrain[middle] = min(max(terrain[middle], low), high)
        self.divide(terrain, start, middle, roughness / 2)
        self.divide(terrain, middle, end, roughness / 2)

    def draw_terrain(self):
        x_scale = int(self.width / self.terrain_size)
        y_scale = int(self.width)

        for i in range(1, len(self.terrain)):
            x1 = (i - 1) * x_scale
            y1 = self.height - int(self.terrain[i - 1] * y_scale)
            x2 = i * x_scale
            y2 = self.height - int(self.terrain[i] * y_scale)
            line = QGraphicsLineItem(x1, y1, x2, y2)
            self.scene.addItem(line)

    def draw_axes(self):
        # X axis
        x_axis = QGraphicsLineItem(0, self.height - 10, self.width, self.height - 10)
        self.scene.addItem(x_axis)
        for i in range(0, self.width, 50):
            label = QGraphicsTextItem(str(i))
            label.setPos(i, self.height - 10)
            self.scene.addItem(label)

        # Y axis
        y_axis = QGraphicsLineItem(2, 0, 2, self.height)
        self.scene.addItem(y_axis)
        for i in range(0, self.height, 50):
            label = QGraphicsTextItem(str(i))
            label.setPos(5, self.height - i)
            self.scene.addItem(label)

    def draw_rocket(self):
        rocket_start_x = self.width / 2 - self.rocket_width / 2
        rocket_start_y = self.rocket_height  # Устанавливаем ракету на нижнюю границу окна
        rocket = QGraphicsPolygonItem()
        rocket.setPolygon(QPolygonF([
            QPointF(rocket_start_x, rocket_start_y),
            QPointF(rocket_start_x - self.rocket_width / 2, rocket_start_y + self.rocket_height),
            QPointF(rocket_start_x + self.rocket_width / 2, rocket_start_y + self.rocket_height)
        ]))
        rocket.setBrush(Qt.red)
        self.scene.addItem(rocket)
        return rocket

    def move_rocket(self, event):
        key = event.key()

        # Получаем текущую позицию ракеты на сцене
        rocket_pos = self.rocket_item.scenePos()
        x = rocket_pos.x() + self.rocket_width / 2
        y = rocket_pos.y() + self.rocket_height / 2

        if key == Qt.Key_Left and x > 0:
            self.rocket_item.moveBy(-5, 0)
        elif key == Qt.Key_Right and x < self.width - self.rocket_width:
            self.rocket_item.moveBy(5, 0)
        elif key == Qt.Key_Up and y > 0:
            self.rocket_item.moveBy(0, -5)
        elif key == Qt.Key_Down and y < self.height - self.rocket_height:
            self.rocket_item.moveBy(0, 5)

        # После перемещения обновим координаты
        rocket_pos = self.rocket_item.scenePos()
        x = rocket_pos.x() + self.rocket_width / 2
        y = rocket_pos.y() + self.rocket_height / 2

        print(f'position: x: {x}, y: {y};')


def main():
    app = QApplication(sys.argv)
    window = RocketLanderApp(10, 0.5, width=1200, height=600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
