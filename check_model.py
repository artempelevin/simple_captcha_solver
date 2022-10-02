import os

import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from captcha import Captcha


def _main():
    x_test = []  # Входные данные для тестирования
    y_test = []  # Выходные данные для тестирования

    capthca = Captcha('model.h5')  # Открываем модель с нейронкой
    answers = open('test_images/answers.txt', 'r')  # Файл с правильными ответами
    good_answers = 0  # Кол-во правильных ответов

    # Пробегаемся по 100 капчам (с 1й по 100ю)
    for i in range(1, 101):
        img = cv2.imread(f'test_images/{i}.png')  # Грузим i-тую капчу
        answer = answers.readline().replace('\n', '')  # Получаем правильный ответ
        net_answer = capthca.solve_captcha(img)  # Получаем ответ сети

        # Если ответы сходятся, прибавляем балл
        if net_answer == answer:
            good_answers += 1
            print(f'{net_answer} ? {answer} +')
        else:
            print(f'{net_answer} ? {answer}')

    print(f'{good_answers} правильных ответов из 100')


if __name__ == '__main__':
    _main()
