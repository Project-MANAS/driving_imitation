import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset

from .constant import *


class BatchContainer(object):
	def __init__(self, dataset, count):
		self.dataset = dataset
		self.count = count

	def build_graph(self):
		self.iterator = self.dataset.make_initializable_iterator()
		self.initialize_iterator = self.iterator.initializer
		self.get_next = self.iterator.get_next()


class BatchContainers(object):
	def __init__(self, batch_containers: [BatchContainer]):
		self.batch_containers = batch_containers
		self.curr_type = None

	def __getitem__(self, item):
		return self.batch_containers[item]

	def build_graph(self):
		for batch_container in self.batch_containers:
			batch_container.build_graph()

	@property
	def curr_batch_container(self) -> BatchContainer:
		return self[self.curr_type]


def read_csv(filename):
	with open(filename, 'r') as f:
		f.readline()
		lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]

		inputs = []
		targets = []
		for line in lines:
			inputs.append(line[0])
			targets.append([float(x) for x in line[1:]])
		return inputs, targets

	# lines = map(lambda x: (x[0], np.float32(x[1:])), lines)  # imagefile, outputs
	# return lines


# TODO Avoid reading csv file twice
inputs, targets = read_csv(filename=DATASET_DIR + "/bag_extraction/interpolated.csv")
mean = np.sum(targets, axis=0) / len(targets)
std = np.sqrt(np.sum(np.square(targets), axis=0) / len(targets))


def process_csv(filename):
	global mean, std
	inputs, targets = read_csv(filename)

	mean = np.sum(targets, axis=0) / len(targets)
	std = np.sqrt(np.sum(np.square(targets), axis=0) / len(targets))

	# leave val% for validation
	overlap_sequences = lambda sequence, seq_size: np.array([sequence[i:i + seq_size] for i in
															 range(0, len(sequence) - seq_size, N_CAMS)])
	input_sequences = overlap_sequences(inputs, LEFT_CONTEXT + SEQ_LEN)
	target_sequences = overlap_sequences(targets, SEQ_LEN)

	return mean, std, input_sequences, target_sequences


def get_datasets(filename=DATASET_DIR + "/bag_extraction/interpolated.csv"):
	mean, std, input_sequences, target_sequences = process_csv(
		filename=filename)  # concatenated interpolated.csv from rosbags

	print(mean, std)
	print(len(input_sequences))

	shuffled_indices = np.random.permutation(len(input_sequences))
	input_sequences = input_sequences[shuffled_indices]
	target_sequences = target_sequences[shuffled_indices]

	def index_to_data(input_indices, target):
		input_episode_images_flat = tf.stack(
			[
				tf.image.decode_png(
					tf.read_file(DATASET_DIR + "/bag_extraction/" + x))
				for x in
				tf.unstack(
					tf.reshape(input_indices, shape=[BATCH_SIZE * (LEFT_CONTEXT + SEQ_LEN)])
				)
			]
		)
		# TODO Perform op at deque input_episode_images = tf.cast(input_episode_images_flat, tf.float32) * 2.0 / 255.0 - 1.0
		input_episode = tf.reshape(input_episode_images_flat,
								   shape=[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS])
		target_normalized = tf.reshape((target - mean) / std, [BATCH_SIZE, SEQ_LEN, OUTPUT_DIM])
		return input_episode, target_normalized

	total_batch_count = len(input_sequences)
	validation_batch_index = int(total_batch_count * (1 - validation_fraction - test_fraction) / BATCH_SIZE)
	test_batch_index = int(total_batch_count * (1 - test_fraction) / BATCH_SIZE)
	dataset = lambda input_seq, target_seq: Dataset.from_tensor_slices((input_seq, target_seq)) \
		.shuffle(BUFFER_SIZE) \
		.batch(BATCH_SIZE)
	train_dataset = dataset(input_sequences, target_sequences).take(validation_batch_index).map(index_to_data)
	validation_dataset = dataset(input_sequences, target_sequences).skip(validation_batch_index).take(
		test_batch_index - validation_batch_index).map(
		index_to_data)
	test_dataset = dataset(input_sequences, target_sequences).skip(test_batch_index).map(index_to_data)

	return mean, std, BatchContainers([BatchContainer(train_dataset, validation_batch_index),
									   BatchContainer(validation_dataset, test_batch_index - validation_batch_index),
									   BatchContainer(test_dataset,
													  int((total_batch_count / BATCH_SIZE) - test_batch_index))])
