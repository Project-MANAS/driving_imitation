from enum import Enum
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2

from .constant import *


class DatasetType(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class Welford:
    def __init__(self):
        self.k = 0
        self.M = 0
        self.S = 1

    def __call__(self, x):
        self.k += 1
        newM = self.M + (x - self.M) / self.k
        newS = self.S + np.multiply(x - self.M, x - newM)
        self.M, self.S = newM, newS

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        return np.sqrt(self.S / (self.k - 1))


def image_normalize(img_filenames):
    metrics_file = Path(DATASET_DIR + "image_metrics.npy")

    if metrics_file.exists():
        metrics = np.load(DATASET_DIR + "image_metrics.npy")
        return metrics[0], metrics[1]

    welford = Welford()

    for img_filename in img_filenames:
        img = cv2.imread(DATASET_DIR + img_filename)
        welford(img)

    mean, std = welford.mean, welford.std
    metrics = np.array([mean, std])
    np.save(DATASET_DIR + "image_metrics.npy", metrics)
    return mean, std


def image_normalize_test(img_filenames):
    images = np.array([cv2.imread(DATASET_DIR + filename) for filename in img_filenames])
    return np.mean(images, axis = 0), np.std(images, axis = 0)


def read_csv(filename):
    with open(filename, 'r') as f:
        f.readline()
        lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
        inputs = []
        targets = []
        for line in lines:
            if 'center' not in line[0]:
                continue
            inputs.append(line[0])
            targets.append([float(x) for x in line[1:]])
        return inputs, np.array(targets, dtype = np.float32)


def process_csv(filename):
    inputs, targets = read_csv(filename)

    mean_targets = tf.constant(np.mean(targets, axis = 0), tf.float32, name = "mean_targets")
    std_targets = tf.constant(np.std(targets, axis = 0), tf.float32, name = "std_targets")

    mean_images, std_images = image_normalize(inputs[0:100])

    mean_images = tf.constant(mean_images, tf.float32, name = "mean_images")
    std_images = tf.constant(std_images, tf.float32, name = "std_images")

    def overlap_sequences(sequence, seq_size):
        return np.array([sequence[i:i + seq_size] for i in range(0, len(sequence) - seq_size)])

    input_sequences = overlap_sequences(inputs, LEFT_CONTEXT + SEQ_LEN)
    target_sequences = overlap_sequences(targets, SEQ_LEN)

    return mean_targets, std_targets, mean_images, std_images, input_sequences, target_sequences


def get_datasets(filename = DATASET_DIR + "interpolated.csv"):
    print("Dataset boi is loading")

    mean_t, std_t, mean_i, std_i, input_sequences, target_sequences = process_csv(filename)

    shuffled_indices = np.random.permutation(len(input_sequences))
    input_sequences = input_sequences[shuffled_indices]
    target_sequences = target_sequences[shuffled_indices]

    def norm_image(img_file):
        img = tf.image.decode_png(tf.read_file(img_file))
        img = tf.cast(img, tf.float32)
        return tf.div(img - mean_i, std_i)

    def index_to_data(input_indices, target):
        input_episode_images_flat = tf.stack(
            [
                norm_image(DATASET_DIR + x)
                for x in
                tf.unstack(
                    tf.reshape(input_indices, shape = [BATCH_SIZE * (LEFT_CONTEXT + SEQ_LEN)])
                )
            ]
        )
        input_episode_images_flat = tf.cast(input_episode_images_flat, tf.float32)

        input_episode = tf.reshape(input_episode_images_flat,
                                   [BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS])
        input_episode = tf.transpose(input_episode, [0, 4, 1, 2, 3])
        target_normalized = tf.reshape(tf.div(target - mean_t, std_t), [BATCH_SIZE, SEQ_LEN, OUTPUT_DIM])

        return input_episode, target_normalized

    total_batch_count = len(input_sequences)
    validation_batch_index = int(total_batch_count * (1 - validation_fraction - test_fraction) / BATCH_SIZE)
    test_batch_index = int(total_batch_count * (1 - test_fraction) / BATCH_SIZE)

    def make_dataset(input_seq, target_seq):
        return tf.data.Dataset().from_tensor_slices((input_seq, target_seq)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train_dataset = make_dataset(input_sequences, target_sequences) \
        .take(validation_batch_index) \
        .map(index_to_data, N_THREADS) \
        .prefetch(BATCH_SIZE * 5)

    validation_dataset = make_dataset(input_sequences, target_sequences) \
        .skip(validation_batch_index) \
        .take(test_batch_index - validation_batch_index) \
        .map(index_to_data, N_THREADS) \
        .prefetch(BATCH_SIZE * 5)

    test_dataset = make_dataset(input_sequences, target_sequences) \
        .skip(test_batch_index) \
        .map(index_to_data, N_THREADS) \
        .prefetch(BATCH_SIZE * 5)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    type_ops = {DatasetType.TRAIN: iterator.make_initializer(train_dataset),
                DatasetType.VALIDATION: iterator.make_initializer(validation_dataset),
                DatasetType.TEST: iterator.make_initializer(test_dataset)}

    no_of_iters = total_batch_count // BATCH_SIZE
    iter_size = {DatasetType.TRAIN: no_of_iters,
                 DatasetType.VALIDATION: int(no_of_iters * validation_fraction),
                 DatasetType.TEST: int(no_of_iters * test_fraction)}

    print("Dataset boi has finished loading")
    return iterator, type_ops, mean_t, std_t, iter_size
