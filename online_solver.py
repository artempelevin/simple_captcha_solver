import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
from tensorflow import keras
from captcha import get_image, remove_line1, \
    remove_line2, to_binary, get_numbers, NUM_OF_IMAGES


def _main() -> None:
    path_to_model = 'model.h5'
    # Открываем натренированную модель нейросети
    model = keras.models.load_model(path_to_model)

    while True:
        # Скачиваем капчу и получаем её в виде np-массива
        img = get_image('https://foiz.ru/reg/captcha.php?1616923356')

        # Отображаем картинку
        plt.subplot(4, 1, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Raw image')
        plt.imshow(img)

        img = remove_line1(img)  # Удаляем линии алгоритмом №1 (удаляет почти всё)
        # Отображаем картинку
        plt.subplot(4, 1, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('Remove lines')
        plt.imshow(img)

        numbers = get_numbers(img)  # Делим капчу на 4 цифры

        answer = ''  # Ответ нейросети на капчу

        plt.subplot(4, 1, 3)
        plt.title('Split into numbers')
        # Проходимся по каждой из цифр
        for i in range(NUM_OF_IMAGES):
            number = numbers[i]  # Получаем i-тую цифру
            # Применяем пороговую функцию к цифре
            _, number = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
            # Отображаем картинку
            plt.subplot(4, 4, 9 + i)
            plt.xticks([])
            plt.yticks([])
            plt.title('Split into numbers')
            plt.imshow(number)
            # Дочищаем остатки от линий (если линия и цифра одного цвета, то не выйдет)
            number = remove_line2(number)
            # Переводим в двоичный формат изображение (только нули и единицы)
            number = to_binary(number)
            # Получаем ответ от нейросети на текущую цифру
            answer += str(model.predict(np.array([number])).argmax())

            # Отображаем цифру
            plt.subplot(4, 4, 13 + i)
            plt.imshow(number)
            plt.xticks([])
            plt.yticks([])
            plt.title('Monochrome')

        print(answer)
        plt.waitforbuttonpress()


if __name__ == '__main__':
    _main()
