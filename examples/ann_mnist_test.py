import numpy as np
from dlimage import Network
from dlimage.mnist import MNISTLoader


def vectorize(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
       This is used to convert a digit into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


mndata = MNISTLoader('../dlimage//mnist/data')
images, labels = mndata.load_training()
training_x = [i.reshape(784, 1) for i in images.copy()]
training_y = [vectorize(i) for i in labels.copy()]
training_data = zip(training_x, training_y)

images, labels = mndata.load_testing()
test_x = [i.reshape(784, 1) for i in images]
test_data = zip(test_x, labels)


net = Network([784, 80, 10])
net.SGD(training_data, 300, 200, 0.995, test_data=test_data)

