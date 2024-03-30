import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RocketLandingSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Rocket Landing Simulation")

        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack()

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.terrain_size = 100
        self.roughness = 1.0

        self.terrain = self.generate_terrain(self.terrain_size, self.roughness)

        self.terrain_line, = self.ax.plot(self.terrain)

        self.rocket = self.ax.plot([], [], 'r')[0]

        self.animation = FuncAnimation(self.figure, self.update, frames=range(100),
                                       repeat=False, interval=100)

        self.canvas_widget = FigureCanvasTkAgg(self.figure, master=self.canvas)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generate_terrain(self, size, roughness):
        terrain = np.zeros(size)
        terrain[0] = np.random.rand()
        terrain[-1] = np.random.rand()

        self.divide(terrain, 0, len(terrain) - 1, roughness)
        return terrain

    def divide(self, terrain, start, end, roughness):
        if end - start < 2:
            return

        middle = (start + end) // 2
        terrain[middle] = (terrain[start] + terrain[end]) / 2 + np.random.randn() * roughness

        self.divide(terrain, start, middle, roughness / 2)
        self.divide(terrain, middle, end, roughness / 2)

    def update(self, frame):
        x = frame
        y = self.terrain[frame]
        self.rocket.set_data(x, y)
        self.ax.set_xlim(0, self.terrain_size)
        self.ax.set_ylim(0, 1)
        return self.rocket,


def main():
    root = tk.Tk()
    app = RocketLandingSimulation(root)
    root.mainloop()


if __name__ == "__main__":
    main()
