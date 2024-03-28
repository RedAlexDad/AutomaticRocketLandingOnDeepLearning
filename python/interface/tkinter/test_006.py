import numpy as np
import tkinter as tk


class RocketLanderApp:
    def __init__(self, master, terrain_size, roughness):
        self.master = master
        self.canvas = tk.Canvas(master, width=1000, height=500)
        self.canvas.pack()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.terrain = self.generate_terrain()

        # Создаем ракету
        self.rocket = self.canvas.create_polygon(250, 470, 240, 490, 260, 490, fill="red")
        self.canvas.bind("<KeyPress>", self.move_rocket)
        self.canvas.focus_set()

        # Создаем график рельефа
        self.plot_terrain()

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
        x, y = self.canvas.coords(self.rocket)[0], self.canvas.coords(self.rocket)[1]
        if key == "Left":
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right":
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up":
            self.canvas.move(self.rocket, 0, -5)
        elif key == "Down":
            self.canvas.move(self.rocket, 0, 5)

        print(f'''position: x: {x}, y: {y}''')

        # Проверка столкновения с поверхностью
        if y <= self.terrain[int(x) // 5] * 500:
            self.check_landing(y, self.terrain[int(x) // 5])

    def check_landing(self, rocket_y, terrain_height):
        if rocket_y >= terrain_height * 500:  # Приземление
            print("Приземление успешно!")
        else:  # Авария
            print("Крах!")

    def plot_terrain(self):
        x_scale = 500 / self.terrain_size
        y_scale = 500
        for i in range(1, len(self.terrain)):
            x1 = (i - 1) * x_scale
            y1 = self.terrain[i - 1] * y_scale
            x2 = i * x_scale
            y2 = self.terrain[i] * y_scale
            self.canvas.create_line(x1, y1, x2, y2, fill="black")


def main():
    root = tk.Tk()
    app = RocketLanderApp(root, 100, 0.1)
    root.mainloop()


if __name__ == "__main__":
    main()


