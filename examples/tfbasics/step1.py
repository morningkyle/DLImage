""" This file creates a basic tensorflow adder component which is then used to calculate the sum of
    its two inputs. """
import tensorflow as tf

# Create an adder component
node1 = tf.constant(3.0, tf.float32, name='const_a')
node2 = tf.constant(5.0, tf.float32, name='const_b')
node3 = tf.add(node1, node2, name='adder')

# Run the model
sess = tf.Session()
v = sess.run(node3)
print("output: ", v)
sess.close()


