import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from datetime import datetime

from dlimage.mnist import MNISTLoader


def vectorize(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e

mndata = MNISTLoader('dlimage/mnist/data')
images, lables = mndata.load_training()
x_train = np.ndarray((len(images), len(images[0])))
y_train = np.ndarray((len(lables), 10))
for i in range(len(images)):
    x_train[i] = images[i]
for i in range(len(lables)):
    y_train[i] = vectorize(lables[i])
print("Loading training data finished.")

mndata = MNISTLoader('dlimage/mnist/data')
images, lables = mndata.load_testing()
x_test = np.ndarray((len(images), len(images[0])))
y_test = np.ndarray((len(lables), 10))
for i in range(len(images)):
    x_test[i] = images[i]
for i in range(len(lables)):
    y_test[i] = vectorize(lables[i])
print("Loading testing data finished.")


model = Sequential()
model.add(Dense(80, activation='sigmoid', input_dim=784))
model.add(Dense(10, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

print("Start training: " + str(datetime.now()))
model.fit(x_train, y_train, epochs=20, batch_size=20, verbose=1)
print("End training: " + str(datetime.now()))

print("Start evaluating: " + str(datetime.now()))
score = model.evaluate(x_test, y_test, batch_size=20)
print(score)
print("End evaluating: " + str(datetime.now()))
