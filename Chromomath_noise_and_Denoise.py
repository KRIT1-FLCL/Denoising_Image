# Импорт необходимых библиотек
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import skimage.restoration
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
root.geometry("1000x720")


#Создание фреймов для размещения виджетов:
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.Y)

#Создание виджетов для отображения изображений:
original_image_label = tk.Label(left_frame, text="Оригинальное изображение")
original_image_label.pack()

original_image_canvas = tk.Canvas(left_frame)
original_image_canvas.pack(fill=tk.BOTH, expand=True)

noisy_image_label = tk.Label(right_frame, text="Зашумленное изображение")
noisy_image_label.pack()

noisy_image_canvas = tk.Canvas(right_frame)
noisy_image_canvas.pack(fill=tk.BOTH, expand=True)

#Создание переменных для хранения изображений в формате PIL и numpy
original_image_pil = None
original_image_np = None
noisy_image_pil = None
noisy_image_np = None

mode_var = tk.IntVar()

def load_image():
    global original_image_pil, original_image_np, noisy_image_pil, noisy_image_np

    #Открытие диалогового окна для выбора файла
    file_name = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Image files", "*.jpg *.png *.bmp")])

    if file_name:

        #Загрузка изображения в формате PIL и numpy
        original_image_pil = Image.open(file_name)
        original_image_np = np.array(original_image_pil)

        #Получение ширины и высоты исходного изображения
        original_width, original_height = original_image_pil.size

        # Получение размера холста для оригинального изображения
        width1 = original_image_canvas.winfo_width()
        height1 = original_image_canvas.winfo_height()

        #Вычисление коэффициента масштабирования
        scale_factor = min(width1 / original_width, height1 / original_height)

        #Вычисление новой ширины и высоты с учетом соотношения сторон
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        #Изменение размера оригинального изображения в формате PIL
        resized_original_image_pil = original_image_pil.resize((new_width, new_height))

        #Отображение оригинального изображения на холсте
        original_image_tk = ImageTk.PhotoImage(resized_original_image_pil)
        original_image_canvas.create_image(width1/2, height1/2, image=original_image_tk)
        original_image_canvas.image = original_image_tk

    #Генерация и отображение зашумленного изображения на холсте
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


# создание функции для изменения типа шума в режиме добавления шума

def change_noise_type(*args):
    global noise_params

    # изменение значения атрибута noise_type словаря noise_params в зависимости
    # от выбранного значения в списке noise_type_listbox
    if noise_params["noise_type"] != "Denoise":
        noise_params["noise_type"] = noise_type_var.get()

    # генерация и отображение зашумленного изображения на холсте
    generate_and_show_noisy_image()


# создание функции для изменения интенсивности шума в режиме добавления шума
# или интенсивности удаления шума в режиме удаления шума/
def change_noise_intensity(value):
    global noise_params

    # изменение значения атрибута noise_intensity или denoise_intensity словаря noise_params в зависимости от положения ползунка noise_intensity_scale
    if noise_params["noise_type"] != "Denoise":
        noise_params["noise_intensity"] = float(value)
    else:
        noise_params["denoise_intensity"] = float(value)


    # генерация и отображение зашумленного изображения на холсте
    #generate_and_show_noisy_image()

def switch_mode():
    global noise_params
    # Получение значения выбранной кнопки с помощью метода get() переменной mode_var
    mode_value = mode_var.get()
    # Сравнение значения с 1 или 2 для определения режима работы программы
    if mode_value == 1:
        # Режим добавления шума
        noise_params["noise_type"] = noise_type_var.get()
    elif mode_value == 2:
        # Режим удаления шума
        noise_params["noise_type"] = "bilateral"
    elif mode_value == 3:
        # Режим удаления шума
        noise_params["noise_type"] = "median"
    elif mode_value == 4:
        # Режим удаления шума
        noise_params["noise_type"] = "nonlocal"


def denoise_image_nonlocal_means(image, intensity):

    # Вычисляем размер патча и степень фильтрации в зависимости от интенсивности удаления шума
    patch_size = int(intensity * 10) + 3
    # Размер патча будет меняться от 3 до 13
    filter_strength = intensity * 20 + 1
    # Степень фильтрации будет меняться от 1 до 21

    # Применяем фильтр Нонлокал-Минс к всему изображению с помощью функции fastNlMeansDenoising() из библиотеки cv2
    # Передаем размер патча и степень фильтрации в качестве аргументов h и templateWindowSize
    denoised_image = cv2.fastNlMeansDenoising(image, h=filter_strength, templateWindowSize=patch_size)

    # Возвращаем обработанное изображение из функции
    return denoised_image

#Создание функции для удаления шума с помощью билатерального фильтра
def denoise_image_bilateral(image, intensity):
    # Вычисляем диаметр окна и сигмы для цвета и пространства в зависимости от интенсивности удаления шума
    diameter = int(intensity * 40) + 5
    # Диаметр окна будет меняться от 5 до 45
    sigma_color = intensity * 150
    # Сигма для цвета будет меняться от 0 до 150
    sigma_space = intensity * 20 # Сигма для пространства будет меняться от 0 до 20

    # Применяем билатеральный фильтр к всему изображению
    denoised_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    # Возвращаем обработанное изображение из функции
    return denoised_image

#Создание функции для удаления шума с помощью медианного фильтра
def denoise_image_median(image, intensity):
    # Вычисляем размер окна в зависимости от интенсивности удаления шума
    kernel_size = int(intensity * 10) + 1
    # Размер окна будет меняться от 1 до 11
    if kernel_size % 2 == 0:
        # Если размер окна четный, то увеличиваем его на 1
        kernel_size += 1

    # Применяем медианный фильтр к всему изображению
    denoised_image = cv2.medianBlur(image, kernel_size)

    # Возвращаем обработанное изображение из функции
    return denoised_image

def generate_gaussian_noise(image, intensity):
    # вычисление стандартного отклонения гауссовского распределения
    # в зависимости от интенсивности шума

    std = intensity * 255

    # создание пустой матрицы того же размера и типа данных, что и исходное изображение
    noise_matrix = np.zeros_like(image)

    # заполнение матрицы случайными значениями по стандартному нормальному распределению
    # с помощью cv2.randn() и умножение на std
    cv2.randn(noise_matrix, 0, std)

    # сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)
    return noisy_image


# создание функции для генерации шума соли и перца с заданной интенсивностью
def generate_salt_and_pepper_noise(image, intensity):
    # генерация матрицы случайных значений от 0 до 1 с размером и типом данных исходного изображения
    # с помощью np.random.rand()
    noise_matrix = np.random.rand(*image.shape)

    # создание копии исходного изображения
    noisy_image = image.copy()

    # замена пикселей изображения на черные, если значение шума меньше интенсивности / 2
    noisy_image[noise_matrix < intensity / 2] = 0

    # замена пикселей изображения на белые, если значение шума больше 1 - интенсивности / 2
    noisy_image[noise_matrix > 1 - intensity / 2] = 255
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

def generate_quantization_noise(image, intensity):
    # вычисление количества уровней яркости в зависимости от интенсивности шума
    # с помощью формулы levels = int(1 / intensity) + 1
    levels = int(1 / intensity) + 1

    # вычисление шага квантования в зависимости от количества уровней яркости
    # с помощью формулы 255 / (levels - 1)

    step = 255 / (levels - 1)

    # округление исходного изображения до ближайшего кратного шага квантования
    # с помощью функции np.round()
    rounded_image = np.round(image / step) * step

    # приведение округленного изображения к целочисленному типу данных np.uint8
    # с помощью метода astype()
    quantized_image = rounded_image.astype(np.uint8)
    return quantized_image


# Создание функции для генерации зернистости пленки с заданной интенсивностью
def generate_film_grain_noise(image, intensity):
    # Вычисление коэффициента контраста в зависимости от интенсивности шума
    contrast = intensity * 255

    # Генерация матрицы случайных значений с размером и типом данных исходного изображения по равномерному распределению с помощью np.random.uniform()
    noise_matrix = np.random.uniform(-contrast, contrast, image.shape).astype(np.uint8)

    # Сложение исходного изображения с матрицей шума поэлементно с учетом переполнения с помощью np.add()
    noisy_image = np.add(image, noise_matrix)
    return noisy_image

def generate_periodic_noise(image, intensity):

    # создание массива с периодическим узором из горизонтальных черных полос
    # можно менять толщину и расстояние между полосами для получения разных узоров
    stripe_width = int(intensity * 10) + 1
    stripe_gap = int(intensity * 20) + 1
    stripe_pattern = np.zeros((stripe_width + stripe_gap, image.shape[1]))
    stripe_pattern[:stripe_width, :] = 125

    # повторение узора по вертикали для заполнения всего изображения
    # с помощью функции np.tile()
    stripe_pattern = np.tile(stripe_pattern, (image.shape[0] // (stripe_width + stripe_gap) + 1, 1))

    # обрезание узора по размеру исходного изображения
    # с помощью оператора среза [:,:]
    stripe_pattern = stripe_pattern[:image.shape[0], :]

    # добавление узора к исходному изображению с помощью оператора +
    # и нормализация результата в диапазоне от 0 до 255
    noisy_image = image + stripe_pattern
    noisy_image = np.clip(noisy_image, 0, 255)
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
            "Denoise": denoise_image,
            "nonlocal": denoise_image_nonlocal_means,
            "median": denoise_image_median,
            "bilateral": denoise_image_bilateral
        }

        # Выбор функции для генерации или удаления шума в зависимости от типа шума из словаря noise_functions
        noise_function = noise_functions[noise_type]


        noisy_image_np = np.zeros_like(original_image_np)
        for i in range(3):
            if noise_type == "Denoise":
                noisy_image_np[:, :, i] = noise_function(original_image_np[:, :, i],denoise_intensity)
            elif noise_type == "Salt and pepper":
                noisy_image_np = noise_function(original_image_np, noise_intensity)
                # break
            else: noisy_image_np[:, :, i] = noise_function(original_image_np[:, :, i], noise_intensity)

        noisy_image_pil = Image.fromarray(noisy_image_np)

        # Получение ширины и высоты исходного изображения
        original_width, original_height = original_image_pil.size

        width2 = noisy_image_canvas.winfo_width()
        height2 = noisy_image_canvas.winfo_height()

        # Вычисление коэффициента масштабирования
        scale_factor = min(width2 / original_width, height2 / original_height)

        # Вычисление новой ширины и высоты с учетом соотношения сторон
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Изменение размера зашумленного изображения в формате PIL
        resized_noisy_image_pil = noisy_image_pil.resize((new_width, new_height))

        # Отображение зашумленного изображения на холсте
        noisy_image_tk = ImageTk.PhotoImage(resized_noisy_image_pil)
        noisy_image_canvas.create_image(width2 / 2, height2 / 2, image=noisy_image_tk)
        noisy_image_canvas.image = noisy_image_tk



        # Оценка качества зашумленного изображения с помощью SSIM и вывод результата на экран
        #ssim_value = ssim(original_image_np, noisy_image_np, multichannel=True)
        ssim_value = ssim(original_image_np, noisy_image_np, multichannel=True, win_size=5, channel_axis=2)
        ssim_label.config(text=f"SSIM: {ssim_value:.4f}")

def denoise_image(image, intensity):
    # Вычисляем диаметр окна и сигмы для цвета и пространства в зависимости от интенсивности удаления шума
    diameter = int(intensity * 40) + 5 # Диаметр окна будет меняться от 5 до 45
    sigma_color = intensity * 150 # Сигма для цвета будет меняться от 0 до 150
    sigma_space = intensity * 20 # Сигма для пространства будет меняться от 0 до 20

    # Применяем билатеральный фильтр к всему изображению
    denoised_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    # Возвращаем обработанное изображение из функции
    return denoised_image


noise_type_label = tk.Label(top_frame, text="Тип шума:")
noise_type_label.pack(side=tk.LEFT)
noise_type_var = tk.StringVar()
noise_type_var.trace("w", change_noise_type)
noise_type_listbox = tk.OptionMenu(top_frame, noise_type_var, "Gaussian", "Salt and pepper", "Shot", "Quantization",
                                   "Film grain", "Periodic")
noise_type_listbox.pack(side=tk.LEFT)
noise_type_var.set("Gaussian")

noise_intensity_label = tk.Label(top_frame, text="Интенсивность:")
noise_intensity_label.pack(side=tk.LEFT)
noise_intensity_scale = tk.Scale(top_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                 command=change_noise_intensity)
noise_intensity_scale.pack(side=tk.LEFT)
noise_intensity_scale.set(0.5)


def apply_noise_reduction():
    generate_and_show_noisy_image()

apply_button = tk.Button(top_frame, text="Применить", command=apply_noise_reduction)
apply_button.pack(side=tk.LEFT)



ssim_label = tk.Label(top_frame, text="SSIM: 0.0000")
ssim_label.pack(side=tk.RIGHT)

load_button = tk.Button(top_frame, text="Загрузить изображение", command=load_image)
load_button.pack(side=tk.RIGHT)

save_button = tk.Button(top_frame, text="Сохранить изображение", command=save_noisy_image)
save_button.pack(side=tk.RIGHT)

mode_label = tk.Label(top_frame, text="Режим работы программы:")
mode_label.pack(side=tk.TOP)

mode_var = tk.IntVar()
mode_switch = tk.Radiobutton(top_frame, text="Добавление шума", value=1, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.TOP)
mode_switch.select()
mode_switch = tk.Radiobutton(top_frame, text="Биратеральный фильтр", value=2, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.TOP)
mode_switch = tk.Radiobutton(top_frame, text="Медианный фильтр", value=3, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.TOP)
mode_switch = tk.Radiobutton(top_frame, text="Фильтр Винера", value=4, variable=mode_var, command=switch_mode)
mode_switch.pack(side=tk.TOP)


# Запуск главного цикла программы
root.mainloop()

# Конец кода

