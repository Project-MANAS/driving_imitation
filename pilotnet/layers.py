import tensorflow as tf


def conv2d(x, w, b, strides=1, padding="SAME"):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv3d(x, w, b, strides=[1, 1, 1, 1, 1], padding="SAME"):
	X = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding="SAME")
	x = tf.nn.bias_add(x, b)
	return tf.relu(x)

def dense(x, w, b):
    x = tf.add(tf.matmul(x, w), b)
    return tf.nn.relu(x)


def flatten(x):
    num_inputs = 1
    for i in x.get_shape().as_list()[1:]:
        num_inputs *= i
    return tf.reshape(x, [-1, num_inputs])
