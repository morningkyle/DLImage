import numpy as np
from dlimage import Network
from dlimage.mnist import MNISTLoader


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
       This is used to convert a digit into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


mndata = MNISTLoader('../dlimage//mnist/data')
images, labels = mndata.load_training()
training_x = [i.reshape(784, 1) for i in images.copy()]
training_y = [vectorized_result(i) for i in labels.copy()]
print(len(training_x), len(training_x))
print(type(training_x[0]))
print(training_x[0].shape)
print(training_x[0].dtype)

print(type(training_y[0]))
print(training_y[0].dtype)
print(training_y[0].shape)
print('------------------')
training_data = zip(training_x, training_y)

images, labels = mndata.load_testing()
test_x = [i.reshape(784, 1) for i in images]
test_y = [vectorized_result(i) for i in labels]
print(len(test_x), len(test_y))
print(type(test_x[0]))
print(test_x[0].shape)
print(test_x[0].dtype)

print(type(test_y[0]))
print(test_y[0].dtype)
print(test_y[0].shape)
test_data = zip(test_x, test_y.copy())


net = Network([784, 10, 10])
net.SGD(training_data, 10, 10, 1.0, test_data=test_data)

