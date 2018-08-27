import sys
import os
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from dlimage.mnist import MNISTLoader


def load_data(data_path):
    mndata = MNISTLoader(data_path)
    images, labels = mndata.load_testing()
    return images, labels


def show_image(image, label):
    plt.title(label, {'fontsize': 36})
    plt.imshow(image.reshape(28, 28), cmap=mpl.cm.Greys)
    plt.show()


def print_data_info(images, labels):
    print("Type of images container: {0}, type of element: {1}".format(type(images), type(images[0])))
    print("Size of image: {0}".format(images[0].shape))
    print("Type of labels container: {0}, type of element: {1}".format(type(labels), type(labels[0])))
    print("Total images: {0}, Total labels: {0}".format(len(images), len(labels)))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = random.randint(0, 10000)

    images, labels = load_data('../dlimage/mnist/data')
    print_data_info(images, labels)
    show_image(images[idx], labels[idx])



