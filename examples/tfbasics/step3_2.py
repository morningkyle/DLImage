import tensorflow as tf

w = tf.Variable([1], dtype=tf.float32, name='w')
b = tf.Variable([1], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, name='input')
y = tf.placeholder(tf.float32, name='output')
model = w * x + b

loss = tf.reduce_mean(tf.constant(0.5, name="0.5") * tf.square(model - y, name='square'), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(train, {x: [1, 2, 3, 4], y: [3, 5, 7, 9]})
print(sess.run([w, b]))

writer = tf.summary.FileWriter('logs', sess.graph)
writer.close()
sess.close()
