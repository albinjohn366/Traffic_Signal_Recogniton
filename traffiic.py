from PIL import Image, ImageFilter
import tensorflow as tf
import os
import numpy as np
from sklearn import model_selection
import sys


# Loading images and labels
def load_data(directory):
    images = []
    labels = []
    for set in os.listdir(directory):
        print('Now in group {}'.format(int(set)))
        for file in os.listdir(os.path.join(directory, set)):
            image = Image.open(os.path.join(os.path.join(directory, set), file))
            image = image.resize((30, 30), Image.ANTIALIAS).filter(
                ImageFilter.SHARPEN)
            data = list(image.getdata())
            images.append(data)
            labels.append(int(set))
    result = (images, labels)
    return result


# Model creation
def model_creation():
    model = tf.keras.Sequential()

    # Adding convolution layers with 32 filters
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(30, 30, 3),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flattening
    model.add(tf.keras.layers.Flatten())

    # Hidden layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    # Compiling model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Main function
def main():
    images, labels = load_data('gtsrb')

    # Splitting data set for training and testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        images, labels, test_size=0.4)

    # Training
    model = model_creation()
    model.fit(x_train, y_train, epochs=10)

    # Testing
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
