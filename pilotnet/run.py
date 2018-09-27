import tensorflow as tf
import model
import os

if __name__ == "__main__":
    with tf.Session() as sess:
        os.environ["train"] = "True"
        model.train(sess)
        model.test(sess)
