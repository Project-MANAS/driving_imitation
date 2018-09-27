import os
import numpy as np
from tfrecord_helper.TFRecordGenerator import TFRecordGenerator
from tfrecord_helper.TFRecordReader import TFRecordReader

# Filenames
image_path = "processed/images/"
tfrecord_name = "driving_imitation"
tfrecord_path = ""
label_filename = "steering_angles.npy"
label_path = "processed/"

def generate_data():
    image_filenames = os.listdir(image_path)

    # Create TFRecord if it doesn't exits
    return TFRecordGenerator(tfrecord_name, "tfrecord/", image_filenames, label_path + label_filename).create()

def load_data():
    return TFRecordReader(tfrecord_name, tfrecord_path)
