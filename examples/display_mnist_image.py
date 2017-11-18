import matplotlib.pyplot as plt
from mnist import MNIST


def load_data():
    mndata = MNIST('../mnist/data', return_type='numpy')
    images, labels = mndata.load_training()
    return images, labels


def display_data_info(images, labels):
    print("type of images container: {0}, type of element: {1}".format(type(images), type(images[0])))
    print("number of images: {0}".format(len(images)))
    print("type of labels container: {0}, type of element: {1}".format(type(labels), type(labels[0])))
    print("number of labels: {0}".format(len(labels)))


if __name__ == '__main__':
    images, labels = load_data()
    display_data_info(images, labels)
    plt.title(labels[10])
    plt.imshow(images[10].reshape(28, 28))
    plt.gray()
    plt.show()


