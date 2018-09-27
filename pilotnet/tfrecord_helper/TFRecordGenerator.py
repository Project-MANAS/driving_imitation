import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class TFRecordGenerator:
    def __init__(self, tfrecord_name, save_path, input_file, output_file):
        self.tfrecord_name = tfrecord_name
        self.save_path = save_path
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create(self):
        labels = np.load(self.output_file)
        dataset = list(zip(self.input_file, labels))
        train, test = train_test_split(dataset, test_size=0.2, shuffle=False)
        for i in range(2):
            if i == 0:
                suffix = "_train"
                dataset = train
            else:
                suffix = "_test"
                dataset = test

            if np.DataSource().exists(self.save_path + self.tfrecord_name + suffix + ".tfrecord"):
                print("TFRecord " + self.tfrecord_name + suffix +  " Exists")
                continue

            writer = tf.python_io.TFRecordWriter(self.save_path + self.tfrecord_name + suffix + ".tfrecord")
            for filename, label in dataset:
                print(filename)
                image = np.load(filename)
                height = image.shape[0]
                width = image.shape[1]
                depth = image.shape[2]
                image = image.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(height),
                    'width': self._int64_feature(width),
                    'depth': self._int64_feature(depth),
                    'image': self._bytes_feature(image),
                    'label': self._float_feature(label)}))
                writer.write(example.SerializeToString())
            writer.close()
            print("Created TFRecord" + self.tfrecord_name + suffix + " Successfully")

        return len(dataset), len(train), len(test)
