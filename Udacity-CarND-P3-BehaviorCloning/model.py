import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from PIL import Image

# env parameters
DATA_PATH = './data/my_data/'
DRIVING_LOG_FILE = 'driving_log.csv'
MODEL_FILE = 'model.h5'
MODEL_WEIGHTS_FILE = 'model_weights.h5'

# data column mapping
STEERING_COL = 3
# steering correction values for three cameras from center, left, right.
STEERING_CORRECTION = [0.0, 0.25, -0.25]

# training parameters
BATCH_SIZE = 32
EPOCHS = 10


def dave2():
    """
    added dropout layers to prevent overfitting as I have limited training data.

    Reference:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    _model = Sequential()
    _model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    _model.add(Lambda(lambda x: (x / 255.0)))
    _model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    _model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    _model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    _model.add(Conv2D(64, (3, 3), activation='relu'))
    _model.add(Conv2D(64, (3, 3), activation='relu'))
    _model.add(Flatten())
    _model.add(Dense(100, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(50, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(10, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(1))
    _model.compile(loss='mse', optimizer='adam')
    return _model


def image_data_generator(img_paths, measurements, batch_size):
    """
    data generator used by Keras to prevent out of memory error 
    when data set size is larger than available memory. 
    It loads batch of images and augment data on the fly.
    """
    batch_index = 0
    images = []
    targets = []
    n = len(img_paths)
    while 1:
        current_index = (batch_index * batch_size) % n
        if n > current_index + batch_size:
            current_batch_size = batch_size
        else:
            current_batch_size = n - current_index
            batch_index = 0

        for i in range(current_batch_size):
            image = np.array(Image.open(img_paths[current_index + i]))
            steering = measurements[current_index + i]
            images.append(image)
            targets.append(steering)

            # argument data
            ## flip
            image = cv2.flip(image, 1)
            images.append(image)
            targets.append(-1.0 * steering)

        batch_index += 1
        yield np.array(images), np.array(targets)
        del images[:]
        del targets[:]


if __name__ == '__main__':
    # Get image paths and steerings
    image_paths = []
    steerings = []
    with open(os.path.join(DATA_PATH, DRIVING_LOG_FILE)) as csvfile:
        reader = csv.reader(csvfile)
        # skip head
        next(reader)

        for row in reader:
            for i in range(3):
                image_paths.append(os.path.join(DATA_PATH, row[i].strip()))
                steerings.append(float(row[STEERING_COL]) + STEERING_CORRECTION[i])

    X_paths_train, X_paths_valid, y_train, y_valid = train_test_split(
        image_paths, steerings, test_size=0.25, random_state=0)
    train_datagen = image_data_generator(X_paths_train, y_train, BATCH_SIZE)
    valid_datagen = image_data_generator(X_paths_valid, y_valid, BATCH_SIZE)

    model = dave2()
    model.summary()
    
    # if trained weights exists, we continue the training on top of it
    if os.path.isfile(MODEL_WEIGHTS_FILE):
        model.load_weights(MODEL_WEIGHTS_FILE)
    model.fit_generator(
        train_datagen, 
        len(X_paths_train) / BATCH_SIZE, 
        validation_data=valid_datagen, 
        validation_steps=len(X_paths_valid) / BATCH_SIZE,
        epochs=EPOCHS)
    model.save_weights(MODEL_WEIGHTS_FILE)
    model.save(MODEL_FILE)
