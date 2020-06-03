from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import cv2


# df_train = pd.read_csv('train.csv', delimiter=',')
# df_test = pd.read_csv('train.csv', delimiter=',')
#
# train_labels = df_train["label"].to_numpy()
# train_images = df_train.drop(columns='label').to_numpy()
# test_labels = df_test["label"].to_numpy()
# test_images = df_test.drop(columns='label').to_numpy()
#
# train_labels = keras.utils.to_categorical(train_labels, 10)
# test_labels = keras.utils.to_categorical(test_labels, 10)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=200,
                    epochs=10, verbose=1)


model.save('model')






