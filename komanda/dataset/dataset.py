from enum import Enum

import numpy as np
import tensorflow as tf

from .constant import *


class DatasetType(Enum):
	TRAIN = 0
	VALIDATION = 1
	TEST = 2


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
		return inputs, targets


def process_csv(filename = INTERPOLATED_CSV_DIR + "interpolated.csv"):
	inputs, targets = read_csv(filename)

	mean = np.sum(targets, axis = 0) / len(targets)
	std = np.sqrt(np.sum(np.square(targets), axis = 0) / len(targets))

	def overlap_sequences(sequence, seq_size):
		return np.array([sequence[i:i + seq_size] for i in range(0, len(sequence) - seq_size, N_CAMS)])

	input_sequences = overlap_sequences(inputs, LEFT_CONTEXT + SEQ_LEN)
	target_sequences = overlap_sequences(targets, SEQ_LEN)

	return mean[0], std[0], input_sequences, target_sequences


def get_datasets(filename = DATASET_DIR + "interpolated.csv"):
	print("Dataset boi is loading")

	mean, std, input_sequences, target_sequences = process_csv(filename = filename)
	# concatenated interpolated.csv from rosbags

	shuffled_indices = np.random.permutation(len(input_sequences))
	input_sequences = input_sequences[shuffled_indices]
	target_sequences = target_sequences[shuffled_indices]

	def index_to_data(input_indices, target):
		input_episode_images_flat = tf.stack(
			[
				tf.image.decode_png(
					tf.read_file(DATASET_DIR + "" + x))
				for x in
				tf.unstack(
					tf.reshape(input_indices, shape = [BATCH_SIZE * (LEFT_CONTEXT + SEQ_LEN)])
				)
			]
		)
		input_episode_images_flat = tf.cast(input_episode_images_flat, tf.float32)

		input_episode = tf.reshape(input_episode_images_flat,
		                           shape = [BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS])
		target_normalized = tf.reshape((target - mean) / std, [BATCH_SIZE, SEQ_LEN, OUTPUT_DIM])

		target_normalized = tf.cast(target_normalized, tf.float32)
		return input_episode, target_normalized

	total_batch_count = len(input_sequences)
	validation_batch_index = int(total_batch_count * (1 - validation_fraction - test_fraction) / BATCH_SIZE)
	test_batch_index = int(total_batch_count * (1 - test_fraction) / BATCH_SIZE)

	def make_dataset(input_seq, target_seq):
		return tf.data.Dataset().from_tensor_slices((input_seq, target_seq)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	train_dataset = make_dataset(input_sequences, target_sequences) \
		.take(validation_batch_index) \
		.map(index_to_data, N_THREADS) \
		.prefetch(BATCH_SIZE * 10)

	validation_dataset = make_dataset(input_sequences, target_sequences) \
		.skip(validation_batch_index) \
		.take(test_batch_index - validation_batch_index) \
		.map(index_to_data, N_THREADS) \
		.prefetch(BATCH_SIZE * 10)

	test_dataset = make_dataset(input_sequences, target_sequences) \
		.skip(test_batch_index) \
		.map(index_to_data, N_THREADS) \
		.prefetch(BATCH_SIZE * 10)

	iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

	type_ops = {DatasetType.TRAIN: iterator.make_initializer(train_dataset),
	            DatasetType.VALIDATION: iterator.make_initializer(validation_dataset),
	            DatasetType.TEST: iterator.make_initializer(test_dataset)}

	no_of_iters = total_batch_count // BATCH_SIZE
	iter_size = {DatasetType.TRAIN: no_of_iters,
	             DatasetType.VALIDATION: int(no_of_iters * validation_fraction),
	             DatasetType.TEST: int(no_of_iters * test_fraction)}

	print("Dataset boi has finished loading")
	return iterator, type_ops, mean, std, iter_size
