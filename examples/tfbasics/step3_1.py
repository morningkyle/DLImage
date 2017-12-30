""" This file trains a line function  y = wx + b with given training data (x_data, y_data), to
    find the proper w and b parameters"""
import tensorflow as tf
import numpy as np

# Create training data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure
w = tf.Variable([1], dtype=tf.float32, name='w')
b = tf.Variable([0.1], dtype=tf.float32, name='b')
y = w * x_data + b
squared_deltas = tf.square(y - y_data, name='squared_deltas') / 2
loss = tf.reduce_mean(squared_deltas, name='loss')
optimizer = tf.train.GradientDescentOptimizer(1)
train = optimizer.minimize(loss)
tf.summary.scalar('loss_summary', loss)
# --- tf structure end ---- #

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


writer = tf.summary.FileWriter('logs', sess.graph)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
for i in range(100):
    summary, _ = sess.run([merged, train], options=run_options)
    writer.add_summary(summary, i)
v = sess.run([w, b])
print("output: ", v)
writer.close()
sess.close()
