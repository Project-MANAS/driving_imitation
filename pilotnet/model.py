import tensorflow as tf
from layers import conv2d, dense, flatten
import matplotlib.pyplot as plt
from tfrecord_helper.TFRecordGenerator import TFRecordGenerator
from tfrecord_helper.TFRecordReader import TFRecordReader
from dataset import generate_data, load_data
import numpy as np
import configuration
import os

# Training Parameters
train = os.environ["train"] == "True"
batch_size = int(os.environ["batch_size"])
num_epochs = int(os.environ["num_epochs"])
display_step = int(os.environ["display_step"])

# Network Parameters
input_size_h = int(os.environ["input_size_h"])
input_size_w = int(os.environ["input_size_w"])
num_channels = int(os.environ["num_channels"])

# Generate and load TFRecord
if train:
    num_samples, num_train_samples, num_test_samples = generate_data()
    tfrecord = load_data()

# TF Graph Input
X = tf.placeholder(tf.float32, shape=[None, input_size_h, input_size_w, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, 1])
mean = tf.constant(np.load("processed/mean.npy"), dtype=tf.float32)
std = tf.constant(np.load("processed/std.npy"), dtype=tf.float32)
# TF Graph
def network(x, weights, biases):
    x = tf.reshape(x, shape=[-1, input_size_h, input_size_w, num_channels])
    x = tf.subtract(x, mean)
    x = tf.divide(x, std)
    x = tf.expand_dims(x, axis=1)
    x = tf.transpose(x, perm=[0, 4, 2, 3, 1])

    conv0 = tf.nn.conv3d(x, weights["wc0"], strides=[1, 1, 1, 1, 1], padding="SAME")
    conv0 = tf.nn.bias_add(conv0, biases["bc0"])
    conv0 = tf.nn.relu(conv0)

    conv0 = tf.transpose(conv0, perm=[0, 1, 4, 2, 3])
    conv0 = tf.reshape(conv0, shape=[-1, 12, 192, 256])
    conv0 = tf.transpose(conv0, perm=[0, 2, 3, 1])

    conv1 = conv2d(conv0, weights["wc1"], biases["bc1"], strides=2)
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"], strides=2)
    conv3 = conv2d(conv2, weights["wc3"], biases["bc3"], strides=2)
    conv4 = conv2d(conv3, weights["wc4"], biases["bc4"], strides=2)
    conv5 = conv2d(conv4, weights["wc5"], biases["bc5"], strides=2)
    conv6 = conv2d(conv5, weights["wc6"], biases["bc6"], strides=2)

    fc1 = flatten(conv6)

    fc1 = dense(fc1, weights["wd1"], biases["bd1"])
    fc2 = dense(fc1, weights["wd2"], biases["bd2"])
    fc3 = dense(fc2, weights["wd3"], biases["bd3"])

    out = tf.add(tf.matmul(fc3, weights["out"]), biases["bias_out"])

    return out


# Parameters
weights = {
    # 5x5 conv, 1 input, 24 outputs
    'wc0': tf.get_variable('wc0', [3, 3, 3, 1, 3], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 5x5 conv, 1 input, 24 outputs
    'wc1': tf.get_variable('wc1', [5, 5, 12, 24], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 5x5 conv, 24 inputs, 36 outputs
    'wc2': tf.get_variable('wc2', [5, 5, 24, 36], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 5x5 conv, 36 inputs, 48 outputs
    'wc3': tf.get_variable('wc3', [5, 5, 36, 48], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 3x3 conv, 48 inputs, 64 outputs
    'wc4': tf.get_variable('wc4', [3, 3, 48, 64], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc5': tf.get_variable('wc5', [3, 3, 64, 64], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc6': tf.get_variable('wc6', [3, 3, 64, 64], tf.float32, tf.contrib.layers.xavier_initializer()),
    # fully connected, 3*4*64 inputs, 100 outputs
    'wd1': tf.get_variable('wd1', [768, 100], tf.float32, tf.contrib.layers.xavier_initializer()),
    # fully connected, 100 inputs, 50 outputs
    'wd2': tf.get_variable('wd2', [100, 50], tf.float32, tf.contrib.layers.xavier_initializer()),
    # fully connected, 50 inputs, 10 outputs
    'wd3': tf.get_variable('wd3', [50, 10], tf.float32, tf.contrib.layers.xavier_initializer()),
    # 10 inputs, 1 output
    'out': tf.get_variable('out', [10, 1], tf.float32, tf.contrib.layers.xavier_initializer())
}

biases = {
    'bc0': tf.get_variable("bc0", [3], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc1': tf.get_variable("bc1", [24], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable("bc2", [36], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable("bc3", [48], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable("bc4", [64], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable("bc5", [64], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bc6': tf.get_variable("bc6", [64], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable("bd1", [100], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable("bd2", [50], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bd3': tf.get_variable("bd3", [10], tf.float32, tf.contrib.layers.xavier_initializer()),
    'bias_out': tf.get_variable("bias_out", [1], tf.float32, tf.contrib.layers.xavier_initializer())
}

# Network Optimization
prediction = network(X, weights, biases)
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
optimizer = tf.train.AdamOptimizer(0.003)
train_op = optimizer.minimize(loss_op)

# Initialize the variables
initialize = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Fetch next batch
if train:
    images_train_batch, labels_train_batch = tfrecord.read(train=True, batch_size=batch_size, num_samples=num_samples, num_epochs=num_epochs)
    images_test_batch, labels_test_batch = tfrecord.read(train=False, batch_size=batch_size)

# Model Saver
saver = tf.train.Saver()

def train(sess):
    sess.run(initialize)

    saver.restore(sess, tf.train.latest_checkpoint('models/'))

    for epochs in range(num_epochs):
        print("Epoch " + str(epochs) + ":")
        for step in range(num_train_samples // batch_size):
            # Fetch next batch
            images_train, labels_train = sess.run((images_train_batch, labels_train_batch))
            # Run optimization (backprop)
            sess.run(train_op, feed_dict={X: images_train, Y: labels_train})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, pred = sess.run([loss_op, prediction], feed_dict={X: images_train, Y: labels_train})
                print("    Step " + str(step) + ", Batch Loss= " + str(loss))

        save_path = saver.save(sess, "models/model")
    print("Optimization Finished")


def test(sess):
    saver.restore(sess, tf.train.latest_checkpoint('models/'))

    for step in range(num_test_samples // batch_size):

        images_test, labels_test = sess.run((images_test_batch, labels_test_batch))

        loss_test, pred_test = sess.run([loss_op, prediction], feed_dict={X: images_test, Y: labels_test})
        # plt.plot(pred_test)
        # plt.plot(labels_test)
        # plt.show()
        print("Test Loss= " + str(loss_test))

def predict(sess, image, restore=True):
    if restore:
        saver.restore(sess, tf.train.latest_checkpoint('models/'))
    pred = sess.run([prediction], feed_dict={X: np.expand_dims(image, axis=0)})
    return pred
