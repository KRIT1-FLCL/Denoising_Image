# Импорт необходимых библиотек
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from collections import deque  # Добавлено для использования deque
from itertools import permutations  # Добавлено для использования permutations

# Создание словаря для хранения параметров шума:
noise_params = {
    "noise_type": "Gaussian",  # Тип шума
    "noise_intensity": 0.1,  # Интенсивность шума
    "denoise_intensity": 0.5  # Интенсивность удаления шума
}

# Создание главного окна программы:
root = tk.Tk()
root.title("Хромоматематическое моделирование дефектов шумовых эффектов в изображениях")
root.geometry("1280x720")

# Создание фреймов для размещения виджетов:
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Создание виджетов для отображения изображений:
original_image_label = tk.Label(left_frame, text="Оригинальное изображение")
original_image_label.pack()

original_image_canvas = tk.Canvas(left_frame, width=400, height=400)
original_image_canvas.pack()

noisy_image_label = tk.Label(right_frame, text="Зашумленное изображение")
noisy_image_label.pack()

noisy_image_canvas = tk.Canvas(right_frame, width=400, height=400)
noisy_image_canvas.pack()

# Создание переменных для хранения изображений в формате PIL и numpy
original_image_pil = None
original_image_np = None
noisy_image_pil = None
noisy_image_np = None

mode_var = tk.IntVar()

# Создание функции для загрузки изображения из файла
def load_image():
    global original_image_pil, original_image_np, noisy_image_pil, noisy_image_np

    # Открытие диалогового окна для выбора файла
    file_name = filedialog.askopenfilename(title="Выберите изображение",
                                           filetypes=[("Image files", "*.jpg *.png *.bmp")])

    if file_name:
        # Загрузка изображения в формате PIL и numpy
        original_image_pil = Image.open(file_name)
        original_image_np = np.array(original_image_pil)

        # Отображение оригинального изображения на холсте
        original_image_tk = ImageTk.PhotoImage(original_image_pil)
        original_image_canvas.create_image(150, 150, image=original_image_tk)
        original_image_canvas.image = original_image_tk

        # Генерация и отображение зашумленного изображения на холсте
        generate_and_show_noisy_image()


# Создание функции для сохранения зашумленного изображения в файл

def save_noisy_image():
    global noisy_image_pil

    if noisy_image_pil:
        # Открытие диалогового окна для выбора имени файла
        file_name = filedialog.asksaveasfilename(title="Сохранить зашумленное изображение",
                                                 filetypes=[("Image files", "*.jpg *.png *.bmp")],
                                                 defaultextension=".jpg")

        if file_name:
            # Сохранение зашумленного изображения в файл
            noisy_image_pil.save(file_name)


# Создание функции для переключения режима работы программы (добавление или удаление шума)
def switch_mode():
    global noise_params

    # Изменение значения атрибута noise_type словаря noise_params в зависимости от положения тумблера mode_switch
    if  mode_var.get() == "Добавление шума":
        noise_params["noise_type"] = noise_type_var.get()
    else:
        noise_params["noise_type"] = "Denoise"

    # Генерация и отображение зашумленного изображения на холсте
    generate_and_show_noisy_image()


# создание функции для изменения типа шума в режиме добавления шума

def change_noise_type(*args):
    global noise_params

    # изменение значения атрибута noise_type словаря noise_params в зависимости
    # от выбранного значения в списке noise_type_listbox
    if mode_var.get() == "Добавление шума":
        noise_params["noise_type"] = noise_type_var.get()


    # if mode_var.get() == "Добавление шума":
    #     noise_params["noise_type"] = noise_functions[3]

    #noise_type


    noise_functions = ("Gaussian","Salt and pepper","Shot","Quantization","Film grain","Periodic","Denoise")

    print(noise_params["noise_type"])
    # генерация и отображение зашумленного изображения на холсте
    generate_and_show_noisy_image()


# создание функции для изменения интенсивности шума в режиме добавления шума
# или интенсивности удаления шума в режиме удаления шума/
def change_noise_intensity(value):
    global noise_params

    # изменение значения атрибута noise_intensity или denoise_intensity словаря noise_params в зависимости от положения ползунка noise_intensity_scale
    if mode_switch == "Добавление шума":
        noise_params["noise_intensity"] = float(value)
    else:
        noise_params["denoise_intensity"] = float(value)

    # генерация и отображение зашумленного изображения на холсте
    generate_and_show_noisy_image()


def switch_mode():
    global noise_params
    # Получение значения выбранной кнопки с помощью метода get() переменной mode_var
    mode_value = mode_var.get()
    # Сравнение значения с 1 или 2 для определения режима работы программы
    if mode_value == 1:
        # Режим добавления шума
        noise_params["noise_type"] = noise_type_var.get()
    else:
        # Режим удаления шума
        noise_params["noise_type"] = "Denoise"

    # Генерация и отображение зашумленного изображения на холсте
    #generate_and_show_noisy_image()

# генерация и отображение зашумленного изображения на холсте
#generate_and_show_noisy_image()


# создание функции для генерации гауссовского шума с заданной интенсивностью
def generate_gaussian_noise(image, intensity):
    # вычисление среднего и стандартного отклонения гауссовского распределения
    # в зависимости от интенсивности шума

    mean = 0
    std = intensity * 255

    # генерация матрицы случайных значений с размером и типом данных исходного изображения
    # по нормальному распределению с помощью np.random.normal()
    noise_matrix = np.random.normal(mean, std, image.shape).astype(np.uint8)

    # сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)

    return noisy_image


# создание функции для генерации шума соли и перца с заданной интенсивностью
def generate_salt_and_pepper_noise(image, intensity):
    # вычисление вероятности появления белого или черного пикселя в зависимости от интенсивности шума

    prob = intensity / 2

    # копирование исходного изображения с помощью np.copy()
    noisy_image = np.copy(image)

    # генерация матрицы случайных значений с размером и типом данных исходного изображения
    # по равномерному распределению с помощью np.random.rand()
    random_matrix = np.random.rand(*image.shape)

    # замена пикселей исходного изображения на белые или черные в зависимости
    # от значения матрицы случайных значений с помощью np.where()
    noisy_image = np.where(random_matrix < prob, 0, noisy_image)
    noisy_image = np.where(random_matrix > 1 - prob, 255, noisy_image)

    return noisy_image


# создание функции для генерации дробового шума с заданной интенсивностью
def generate_shot_noise(image, intensity):
    # вычисление параметра Пуассоновского распределения в зависимости от интенсивности шума
    lam = intensity * 255

    # генерация матрицы случайных значений с размером и типом данных исходного изображения по Пуассоновскому распределению с помощью np.random.poisson()
    noise_matrix = np.random.poisson(lam, image.shape).astype(np.uint8)

    # сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)

    return noisy_image


# создание функции для генерации шума квантования с заданной интенсивностью
def generate_quantization_noise(image, intensity):
    # Вычисление количества уровней квантования в зависимости от интенсивности шума
    levels = int(256 / intensity)

    # Квантование исходного изображения с заданным количеством уровней с помощью np.floor_divide()
    noisy_image = np.floor_divide(image, intensity) * intensity

    # Преобразование типа данных изображения в uint8 с помощью np.astype()
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# Создание функции для генерации зернистости пленки с заданной интенсивностью
def generate_film_grain_noise(image, intensity):
    # Вычисление коэффициента контраста в зависимости от интенсивности шума
    contrast = intensity * 255

    # Генерация матрицы случайных значений с размером и типом данных исходного изображения по равномерному распределению с помощью np.random.uniform()
    noise_matrix = np.random.uniform(-contrast, contrast, image.shape).astype(np.uint8)

    # Сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)

    return noisy_image


# Создание функции для генерации преиодического шума с заданной интенсивностью
def generate_periodic_noise(image, intensity):
    # Вычисление частоты и амплитуды периодического шума в зависимости от интенсивности шума
    frequency = intensity * 10
    amplitude = intensity * 255

    # Получение высоты и ширины исходного изображения
    height, width = image.shape[:2]

    # Создание матрицы координат пикселей изображения с помощью np.arange() и np.meshgrid()
    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)

    # Генерация матрицы значений периодической функции (например, синусоиды) с заданной частотой и амплитудой с помощью np.sin()
    noise_matrix = amplitude * np.sin(2 * np.pi * frequency * xv / width).astype(np.uint8)

    # Сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)

    return noisy_image


# Создание функциии для генерации и отображения зашумленного изображения на холсте
def generate_and_show_noisy_image():
    global original_image_np, noisy_image_pil, noisy_image_np

    if original_image_np is not None:
        # Получение типа и интенсивности шума из словаря noise_params
        noise_type = noise_params["noise_type"]
        noise_intensity = noise_params["noise_intensity"]
        denoise_intensity = noise_params["denoise_intensity"]

        # Создание словаря для хранения функций для генерации или удаления шума в зависимости от типа шума
        noise_functions = {
            "Gaussian": generate_gaussian_noise,
            "Salt and pepper": generate_salt_and_pepper_noise,
            "Shot": generate_shot_noise,
            "Quantization": generate_quantization_noise,
            "Film grain": generate_film_grain_noise,
            "Periodic": generate_periodic_noise,
            "Denoise": denoise_image
        }

        # Выбор функции для генерации или удаления шума в зависимости от типа шума из словаря noise_functions
        noise_function = noise_functions[noise_type]

        # Применение функции для генерации или удаления шума к каждому цветовому каналу исходного изображения
        noisy_image_np = np.zeros_like(original_image_np)
        for i in range(3):
            if noise_type == "Denoise":
                noisy_image_np[:, :, i] = noise_function(original_image_np[:, :, i], denoise_intensity)
            else:
                noisy_image_np[:, :, i] = noise_function(original_image_np[:, :, i], noise_intensity)

        # Преобразование зашумленного изображения в формат PIL
        noisy_image_pil = Image.fromarray(noisy_image_np)

        # Отображение зашумленного изображения на холсте
        noisy_image_tk = ImageTk.PhotoImage(noisy_image_pil)
        noisy_image_canvas.create_image(150, 150, image=noisy_image_tk)
        noisy_image_canvas.image = noisy_image_tk

        # Оценка качества зашумленного изображения с помощью SSIM и вывод результата на экран
        #ssim_value = ssim(original_image_np, noisy_image_np, multichannel=True)
        ssim_value = ssim(original_image_np, noisy_image_np, multichannel=True, win_size=5, channel_axis=2)
        ssim_label.config(text=f"SSIM: {ssim_value:.4f}")


#Создание функции для удаления шума с заданной интенсивностью
def denoise_image(image, intensity):
    # Выбор типа фильтра для удаления шума в зависимости от интенсивности удаления шума
    if intensity < 0.25:
        filter_type = cv2.MEDIAN
    # Медианный фильтр
    elif intensity < 0.5:
        filter_type = cv2.GAUSSIAN  # Гауссовский фильтр
    else:
        filter_type = cv2.BILATERAL  # Билатеральный фильтр

    # Вычисление размера ядра фильтра в зависимости от интенсивности удаления шума
    kernel_size = int(intensity * 20) + 1

    # Применение фильтра к исходному изображению с помощью cv2.blur()
    denoised_image = cv2.blur(image, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT, filterType=filter_type)

    return denoised_image



# Создание виджетов для управления параметрами шума
mode_label = tk.Label(top_frame, text="Режим работы программы:")
mode_label.pack(side=tk.LEFT)

mode_var = tk.IntVar()
mode_switch = tk.Radiobutton(top_frame, text="Добавление шума", value=1, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.LEFT)
mode_switch.select()
mode_switch = tk.Radiobutton(top_frame, text="Удаление шума", value=2, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.LEFT)

# noise_type_label = tk.Label(top_frame, text="Тип шума:")
# noise_type_label.pack(side=tk.LEFT)
# noise_type_var = tk.StringVar()
# noise_type_var.trace("w", change_noise_type)
# noise_type_listbox = tk.OptionMenu(top_frame, noise_type_var, "Gaussian", "Salt and pepper", "Shot", "Quantization",
#                                    "Film grain", "Periodic")
# noise_type_listbox.pack(side=tk.LEFT)
# noise_type_var.set("Salt and pepper")


noise_type_label = tk.Label(top_frame, text="Тип шума:")
noise_type_label.pack(side=tk.LEFT)
noise_type_var = tk.StringVar()
noise_type_var.trace("w", change_noise_type)
noise_type_spinbox = tk.Spinbox(top_frame, textvariable=noise_type_var, values=("Gaussian", "Salt and pepper", "Shot", "Quantization",
                                   "Film grain", "Periodic"), wrap=True)
noise_type_spinbox.pack(side=tk.LEFT)
noise_type_var.set("Gaussian")


noise_intensity_label = tk.Label(top_frame, text="Интенсивность шума:")
noise_intensity_label.pack(side=tk.LEFT)
noise_intensity_scale = tk.Scale(top_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 command=change_noise_intensity)
noise_intensity_scale.pack(side=tk.LEFT)
noise_intensity_scale.set(0.1)

ssim_label = tk.Label(top_frame, text="SSIM: 0.0000")
ssim_label.pack(side=tk.RIGHT)

load_button = tk.Button(top_frame, text="Загрузить изображение", command=load_image)
load_button.pack(side=tk.RIGHT)

save_button = tk.Button(top_frame, text="Сохранить зашумленное изображение", command=save_noisy_image)
save_button.pack(side=tk.RIGHT)

# Запуск главного цикла программы
root.mainloop()

# Конец кода

