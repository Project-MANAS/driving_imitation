import numpy as np
import tensorflow as tf


class TFRecordReader:
    def __init__(self, tfrecord_name, load_path):
        self.tfrecord_name = tfrecord_name
        self.load_path = load_path

    def decode(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.float32)
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.float32)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)

        image_shape = tf.stack([height, width, depth])
        label_shape = tf.stack([1])

        image = tf.reshape(image, image_shape)
        label = tf.reshape(label, label_shape)
        return image, label

    def read(self, train, batch_size=1, num_samples=1, num_epochs=1):
        if train and np.DataSource().exists(self.tfrecord_name + "_train" + ".tfrecord"):
            tfrecord = self.tfrecord_name + "_train" + ".tfrecord"
        else:
            tfrecord = self.tfrecord_name + "_test" + ".tfrecord"

        dataset = tf.data.TFRecordDataset([self.load_path + tfrecord])
        dataset = dataset.map(self.decode)
        if train:
            dataset = dataset.shuffle(num_samples)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()
