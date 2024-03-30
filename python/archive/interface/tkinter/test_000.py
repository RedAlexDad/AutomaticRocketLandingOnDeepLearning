import tkinter as tk
from tkinter import messagebox

# Функция, которая будет вызываться при нажатии на кнопку
def train_and_predict():
    # Здесь вы можете вызвать вашу функцию train_model() и predict()
    # Например:
    model_path = "model.pth"
    # epochs = 10
    # train_model(model_path, epochs)
    # data = [0.5, 0.6, 0.2, 0.8]
    # prediction = predict(data)

    # Выводим результат в диалоговом окне
    messagebox.showinfo("Prediction", f"Prediction: {model_path}")


# Создаем графический интерфейс
root = tk.Tk()
root.title("Automatic Rocket Landing")

# Добавляем метку с описанием
label = tk.Label(root, text="Click the button to train the model and make a prediction.")
label.pack(pady=10)

# Добавляем кнопку
button = tk.Button(root, text="Train and Predict", command=train_and_predict)
button.pack(pady=10)

# Запускаем главный цикл обработки событий
root.mainloop()