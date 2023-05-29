# Импортируем необходимые модули
import cv2 # для работы с изображениями
import numpy as np # для работы с массивами
from matplotlib import pyplot as plt # для визуализации

# Загружаем изображение в формате RGB

#image = cv2.imread(r"high_quality_noise_image.jpg") #10 | 10
image = cv2.imread(r"noisy_image.jpg") # 15 | 80
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Разделяем изображение на цветовые каналы
r, g, b = cv2.split(image)

# Получаем от пользователя диаметр окна и сигмы для цвета и пространства
# (Если диаметр слишком большой, то фильтр может размыть края и детали изображения.
# Если диаметр слишком маленький, то фильтр может не убрать весь шум.)

# Если сигма слишком большая, то фильтр может потерять цветовые переходы и контрастность изображения.
# Если сигма слишком маленькая, то фильтр может не убрать шум в однородных областях.

# Применяем билатеральный фильтр к каждому каналу
# Указываем параметры, полученные от пользователя
r_filtered = cv2.bilateralFilter(r, 25, 75, 75)
g_filtered = cv2.bilateralFilter(g, 25, 75, 75)
b_filtered = cv2.bilateralFilter(b, 25, 75, 75)

# Объединяем отфильтрованные каналы в одно изображение
image_filtered = cv2.merge([r_filtered, g_filtered, b_filtered])

# Оцениваем качество обработанного изображения по метрике SSIM
# Эта метрика измеряет структурное сходство между двумя изображениями
# и принимает значения от -1 до 1, где 1 означает полное совпадение
from skimage.metrics import structural_similarity as ssim
score = ssim(image, image_filtered, multichannel=True, win_size=3)
print(f'SSIM score: {score:.4f}')

# Выводим исходное и обработанное изображения на экран
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Noisy image')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_filtered)
plt.title('Denoised image')
plt.axis('off')
plt.show()


