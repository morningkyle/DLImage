""" This file creates a line function  y = wx + b with given w and b. It is used to calculate the result
    of the input x list. """
import tensorflow as tf

w = tf.Variable([1], dtype=tf.float32, name='w')
b = tf.Variable([1.5], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, name='x')
y = w * x + b

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
v = sess.run(y, {x: [1, 2, 3, 4, 5, 6]})
sess.close()
print("output: ", v)


