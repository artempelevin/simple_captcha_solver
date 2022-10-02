import os

import requests
import numpy as np

import cv2
from typing import List
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import Counter

NUM_OF_IMAGES = 4  # Кол-во изображений с цифрами (всегда 4!)

KERNEL_SIZE = 4  # Размер ядра, ищущего линии (изменяемый параметр)
OFFSET = 3  # Смешение ядра, ищущего линии (изменяемый параметр)


# Скачиваем капчу и возвращаем её как изображение
# Принимает url по которому качает капчу!
def get_image(url: str) -> np.ndarray:
    request = requests.get(url)
    with open('captcha.png', 'wb') as file:
        file.write(request.content)

    img = cv2.imread('captcha.png')
    return img


# Делим капчу на четыре цифры
def get_numbers(img: np.ndarray) -> List[np.ndarray]:
    w = img.shape[1] // NUM_OF_IMAGES  # Ширина одной картинки
    numbers = []
    for i in range(NUM_OF_IMAGES):
        numbers.append(img[:, i * w: i * w + w])

    return numbers


# Удаляем линии с изображения капчи (основывает на том факте, что линии всегда одного цвета, 
# без переходных цветов) и возвращаем новое, изменённое изображение
def remove_line1(img: np.ndarray) -> np.ndarray:
    new_img = img.copy()
    for row in range(0, new_img.shape[0] - KERNEL_SIZE, OFFSET):
        for col in range(0, new_img.shape[1] - KERNEL_SIZE, OFFSET):
            conv = new_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE]  # Полученное ядро

            # В каждом пикселе ядра находим сумму каналов r + g + b
            colors = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
            for i in range(KERNEL_SIZE):
                for j in range(KERNEL_SIZE):
                    colors[i, j] = int(conv[i, j, 0]) + int(conv[i, j, 1]) + \
                                   int(conv[i, j, 2])

            # Получаем список цветов в ядре
            colors = list(Counter(colors.reshape(KERNEL_SIZE ** 2)))
            # Если цветов два и один из них белый - перед нами кусок прямой -> удаляем его
            if len(colors) == 2 and (colors[0] == 765 or colors[1] == 765):
                new_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE] = [255, 255, 255]
    return new_img


# Удаляем остатки линий (различаем их по цвету) с изображения цифры
# и возвращаем новое, изменённое изображение цифры
def remove_line2(number_image: np.ndarray) -> np.ndarray:
    new_img = cv2.cvtColor(number_image, cv2.COLOR_BGR2HSV)  # Переводим из BGR в HSV

    # Каждый пиксель картинки теперь есть сумма его каналов: h + s + v
    colors = np.zeros((new_img.shape[0], new_img.shape[1]), dtype=int)
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            colors[row, col] = int(new_img[row, col, 0]) + int(new_img[row, col, 1]) + \
                               int(new_img[row, col, 2])

            # Получаем все цвета на картинке
    colors = Counter(colors.reshape(colors.shape[0] * colors.shape[1]))
    # Убираем белый цвет
    colors.pop(255)
    # Убираем второй по популярности после белого цвет (по сути - он и есть наша цифра)
    if len(colors) > 0:
        colors.pop(max(colors, key=colors.get))

    # Проходимся по картинке и удаляем все оставшиеся цвета 
    # (как правило - это остатки прямых, но может не сработать, 
    # если прямая и цифра одного цвета)
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            for color in colors:
                if int(new_img[row, col, 0]) + int(new_img[row, col, 1]) + \
                        int(new_img[row, col, 2]) == color:
                    new_img[row, col] = [0, 0, 255]

    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)  # Конвертируем назад в BGR


# Перевод картинку из 3х цветовых слоёв в один цветовой слой, где
# 0 - пиксель, отвечающий фону;
# 1 - пиксель, отвечающий цифре;
def to_binary(img: np.ndarray) -> np.ndarray:
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Если пиксель белый, то это фон -> 0
            if (img[row, col] == [255, 255, 255]).all():
                new_img[row, col] = 0
            # Иначе это цифра -> 1
            else:
                new_img[row, col] = 1

    return new_img


# Класс, который открывает модель и решает капчу
class Captcha:
    # Конструктор класса
    def __init__(self, path_to_model: str) -> None:
        # Открываем натренировонную модель нейросети
        self.model = keras.models.load_model(path_to_model)

        # Решает капчу. Принимает на вход (еобработанное!) изображение капчи!

    def solve_captcha(self, img: np.ndarray) -> str:
        answer = ''  # Ответ нейросети на капчу

        img = remove_line1(img)  # Удаляем линии алгоритмом №1 (удаляет почти всё)
        numbers = get_numbers(img)  # Делим капчу на 4 цифры

        # Проходимся по каждой из цифр
        for i in range(NUM_OF_IMAGES):
            number = numbers[i]  # Получаем i-тую цифру
            # Применяем пороговую функцию к цифре
            _, number = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
            # Дочищаем остатки от линий (если линия и цифра одного цвета, то не выйдет)
            number = remove_line2(number)
            # Переводим в двоичный формат изображение (только нули и единицы)
            number = to_binary(number)

            # Получаем ответ от нейросети на текущую цифру
            answer += str(self.model.predict(np.array([number]), verbose=0).argmax())

        return answer