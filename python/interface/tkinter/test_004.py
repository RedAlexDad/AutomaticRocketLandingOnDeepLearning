import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


class RocketLanderApp:
    def __init__(self, master, terrain_size, roughness):
        self.master = master
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.terrain = self.generate_terrain()

        # Определяем начальные координаты ракеты
        rocket_start_x = 0
        rocket_start_y = self.terrain[int(self.terrain_size / 2)] * 100 - 10  # Положение ракеты по центру поверхности
        self.rocket = self.canvas.create_oval(rocket_start_x, rocket_start_y, rocket_start_x + 20, rocket_start_y + 20,
                                              fill="red")

        self.master.bind("<KeyPress>", self.move_rocket)
        self.canvas.focus_set()

        # Создаем график рельефа
        self.fig, self.ax = plt.subplots()
        self.plot_terrain()
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.BOTTOM)

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
        x, y, _, _ = self.canvas.coords(self.rocket)
        print(f'''position: x: {x}, y: {y}''')
        if key == "Left":
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right":
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up":
            self.canvas.move(self.rocket, 0, 5)
        elif key == "Down":
            self.canvas.move(self.rocket, 0, -5)

        # Проверка столкновения с поверхностью
        if y + 20 >= self.terrain[int(x) // 5]:
            self.check_landing(y, self.terrain[int(x) // 5])

    def check_landing(self, rocket_y, terrain_height):
        if rocket_y >= terrain_height:  # Приземление
            print("Приземление успешно!")
        else:  # Авария
            print("Крах!")

    def plot_terrain(self):
        self.ax.clear()
        self.ax.plot(self.terrain)
        self.ax.set_title('Случайный рельеф')
        self.ax.set_xlabel('Положение')
        self.ax.set_ylabel('Высота')


def main():
    root = tk.Tk()
    app = RocketLanderApp(root, 100, 10)
    root.mainloop()


if __name__ == "__main__":
    main()
