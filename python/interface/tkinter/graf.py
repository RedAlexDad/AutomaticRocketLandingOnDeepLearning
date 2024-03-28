import tkinter as tk


def draw_grid(canvas, step):
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    # рисуем вертикальные линии
    for i in range(0, width, step):
        canvas.create_line(i, 0, i, height, fill="lightgray", tags="grid_line")

    # рисуем горизонтальные линии
    for i in range(0, height, step):
        canvas.create_line(0, i, width, i, fill="lightgray", tags="grid_line")


def draw_axes(canvas):
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    # рисуем ось X
    canvas.create_line(0, height // 2, width, height // 2, fill="black", arrow=tk.LAST)
    # рисуем ось Y
    canvas.create_line(width // 2, 0, width // 2, height, fill="black", arrow=tk.LAST)


def draw_graph(canvas):
    # Рисуем график на canvas
    # В этом примере добавлен просто случайный график
    canvas.create_line(50, 50, 200, 200, fill="blue", width=2)


def main():
    root = tk.Tk()
    root.title("Сетчатый график")

    canvas = tk.Canvas(root, width=400, height=300, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)

    draw_grid(canvas, 20)  # Рисуем сетку с шагом 20 пикселей
    draw_axes(canvas)  # Рисуем оси координат
    draw_graph(canvas)  # Рисуем график

    root.mainloop()


if __name__ == "__main__":
    main()
