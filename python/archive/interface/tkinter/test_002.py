import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk


class RocketLanderApp:
    def __init__(self, master, terrain_size, roughness):
        self.master = master
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.terrain_size = terrain_size
        self.roughness = roughness
        self.terrain = self.generate_terrain()
        self.rocket = self.canvas.create_oval(250, 20, 270, 40, fill="red")  # Создаем ракету
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
        x, y, _, _ = self.canvas.coords(self.rocket)
        if key == "Left":
            self.canvas.move(self.rocket, -5, 0)
        elif key == "Right":
            self.canvas.move(self.rocket, 5, 0)
        elif key == "Up":
            self.canvas.move(self.rocket, 0, -5)
        elif key == "Down":
            self.canvas.move(self.rocket, 0, 5)
        # Проверка столкновения с поверхностью
        if y + 20 >= self.terrain[int(x) // 5]:
            self.check_landing(y, self.terrain[int(x) // 5])

    def check_landing(self, rocket_y, terrain_height):
        if rocket_y >= terrain_height * 100:  # Приземление
            print("Приземление успешно!")
        else:  # Авария
            print("Крах!")


root = tk.Tk()
app = RocketLanderApp(root, 100, 10)
root.mainloop()
