import multiprocessing
import os
from random import shuffle
from typing import Tuple, Optional, List

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Functional
from tensorflow import keras

from captcha import to_binary


def _provide_data(for_train_percent: int, num_of_threads: Optional[int] = 20) -> Tuple[np.ndarray, np.ndarray,
                                                                                       np.ndarray, np.ndarray]:
    assert 1 < for_train_percent < 99
    paths_to_images = []
    for dir_ in range(0, 10):
        paths_to_images.extend(f'dataset/{dir_}/{image_name}' for image_name in os.listdir(f'dataset/{dir_}'))
    shuffle(paths_to_images)

    size = len(paths_to_images)
    train_size = int(size * for_train_percent / 100)
    test_size = size - train_size
    chunk_size = size // num_of_threads

    manager = multiprocessing.Manager()
    data = manager.list()
    processes = []
    for i in range(chunk_size):
        start = i * chunk_size
        stop = (i + 1) * chunk_size
        process = multiprocessing.Process(target=_provide_images, args=(paths_to_images[start:stop], data))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    all_images = [value[0] for value in data]
    all_labels = [value[1] for value in data]
    x_train, y_train = all_images[:train_size], all_labels[:train_size]
    x_test, y_test = all_images[-test_size:], all_labels[-test_size:]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def _provide_images(paths_to_image: List[str], all_images: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    for path_to_image in paths_to_image:
        digit = int(path_to_image.split('/')[-2])
        digit_vectorize = np.zeros(shape=(10,))
        digit_vectorize[digit] = 1

        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        img = to_binary(img)
        all_images.append((img, digit_vectorize))


def _provide_model() -> Functional:
    input_ = keras.layers.Input(shape=(64, 45))

    net = keras.layers.Reshape((64, 45, 1))(input_)
    net = keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', use_bias=True)(net)
    net = keras.layers.MaxPooling2D((2, 2))(net)

    net = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', use_bias=True)(net)
    net = keras.layers.MaxPooling2D((2, 2))(net)

    net = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', use_bias=True)(net)
    net = keras.layers.MaxPooling2D((2, 2))(net)

    net = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', use_bias=True)(net)
    net = keras.layers.MaxPooling2D((2, 2))(net)
    net = keras.layers.Dropout(0.2)(net)

    net = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', use_bias=True)(net)
    net = keras.layers.MaxPooling2D((2, 2))(net)
    net = keras.layers.Reshape((2 * 1 * 256,))(net)
    net = keras.layers.Dropout(0.8)(net)

    output = keras.layers.Dense(10, activation='softmax')(net)
    model = keras.models.Model(inputs=input_, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _main():
    x_train, y_train, x_test, y_test = _provide_data(for_train_percent=80)
    model = _provide_model()
    model.fit(x_train,
              y_train,
              epochs=30,
              validation_data=(x_test, y_test),
              callbacks=[ModelCheckpoint(filepath="weights/{val_accuracy:.5f}.h5", monitor='val_accuracy')])
    model.save('model.h5')


if __name__ == '__main__':
    _main()
