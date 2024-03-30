import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Функция для генерации случайного рельефа
def generate_terrain(size, roughness):
    terrain = np.zeros(size)
    terrain[0] = np.random.rand()
    terrain[-1] = np.random.rand()

    divide(terrain, 0, len(terrain) - 1, roughness)

    return terrain


def divide(terrain, start, end, roughness):
    if end - start < 2:
        return

    middle = (start + end) // 2
    terrain[middle] = (terrain[start] + terrain[end]) / 2 + np.random.randn() * roughness

    divide(terrain, start, middle, roughness / 2)
    divide(terrain, middle, end, roughness / 2)


# Функция для обновления графика в окне Tkinter
def update_plot():
    size = int(size_entry.get())
    roughness = float(roughness_entry.get())

    terrain = generate_terrain(size, roughness)
    ax.clear()
    ax.plot(terrain)
    ax.set_title('Случайный рельеф')
    ax.set_xlabel('Положение')
    ax.set_ylabel('Высота')
    canvas.draw()


# Создание основного окна Tkinter
root = tk.Tk()
root.title('Генератор рельефа')

# Создание и настройка элементов интерфейса
size_label = tk.Label(root, text='Размер:')
size_label.pack()
size_entry = tk.Entry(root)
size_entry.pack()

roughness_label = tk.Label(root, text='Шероховатость:')
roughness_label.pack()
roughness_entry = tk.Entry(root)
roughness_entry.pack()

generate_button = tk.Button(root, text='Сгенерировать', command=update_plot)
generate_button.pack()

# Создание графика в окне Tkinter
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Запуск основного цикла Tkinter
tk.mainloop()
