import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('W_summary', W)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram('b_summary', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, W), biases, name='Wx_plus_b')
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('outputs', outputs)
    return outputs


def generate_training_data():
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    return x_data, y_data


def construct_network():
    # Define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # add hidden layer
    layer_h = add_layer(xs, 1, 10, 'hidden_layer', activation_function=tf.nn.relu)
    # add output layer
    layer_o = add_layer(layer_h, 10, 1, 'output_layer', activation_function=None)

    # the error between prediction and real data
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer_o), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    return xs, ys, train


def initialize_network():
    merged = tf.summary.merge_all()
    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess, writer, merged


def main():
    x_data, y_data = generate_training_data()
    xs, ys, train = construct_network()
    sess, writer, merged = initialize_network()
    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)
    sess.close()


if __name__ == '__main__':
    main()


