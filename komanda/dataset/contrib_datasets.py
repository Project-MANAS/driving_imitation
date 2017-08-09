import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from .constant import *


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


def process_csv(filename, val=5):
	# sum_f = np.float128([0.0] * OUTPUT_DIM)
	# sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
	inputs, targets = read_csv(filename)

	mean = np.sum(targets, axis=0) / len(targets)
	std = np.sqrt(np.sum(np.square(targets), axis=0) / len(targets))

	# leave val% for validation
	overlap_sequences = lambda sequence, seq_size: np.array([sequence[i:i + seq_size] for i in
															 range(0, len(sequence) - seq_size, N_CAMS)])
	input_sequences = overlap_sequences(inputs, LEFT_CONTEXT + SEQ_LEN)
	target_sequences = overlap_sequences(targets, SEQ_LEN)

	# fused_dataset = np.array([[np.reshape(inputs[i:i + LEFT_CONTEXT + SEQ_LEN], [-1, 1]), targets[i:i + LEFT_CONTEXT + SEQ_LEN]] for i in
	# 						  range(0, len(inputs) - (LEFT_CONTEXT + SEQ_LEN), N_CAMS)])

	# cnt = 0
	# for (input, target) in zip(inputs, targets):
	# 	if cnt < SEQ_LEN * BATCH_SIZE * (100 - val):
	# 		train_seq.append(ln)
	# 		sum_f += ln[1]
	# 		sum_sq_f += ln[1] * ln[1]
	# 	else:
	# 		valid_seq.append(ln)
	# 	cnt += 1
	# 	cnt %= SEQ_LEN * BATCH_SIZE * 100
	# mean = sum_f / len(train_seq)
	# var = sum_sq_f / len(train_seq) - mean * mean
	# std = np.sqrt(var)
	# # print(len(train_seq), len(valid_seq))
	# # print(mean, std)  # we will need these statistics to normalize the outputs (and ground truth inputs)
	# return train_seq, valid_seq, mean, std

	return mean, std, input_sequences, target_sequences


validation_batch_index = test_batch_index = total_batch_count = 0


def get_datasets(filename=DATASET_DIR + "/bag_extraction/interpolated.csv"):
	global validation_batch_index, test_batch_index, total_batch_count
	mean, std, input_sequences, target_sequences = process_csv(
		filename=filename,
		val=5)  # concatenated interpolated.csv from rosbags

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
		input_episode_images = tf.cast(input_episode_images_flat, tf.float32) * 2.0 / 255.0 - 1.0
		input_episode = tf.reshape(input_episode_images,
								   shape=[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS])
		target_normalized = (target - mean) / std
		return input_episode, target_normalized

	total_batch_count = len(input_sequences)
	validation_batch_index = int(total_batch_count * (1 - validation_fraction - test_fraction) / BATCH_SIZE)
	test_batch_index = int(total_batch_count * (1 - test_fraction) / BATCH_SIZE)
	dataset = Dataset.from_tensor_slices((input_sequences, target_sequences)) \
		.shuffle(BUFFER_SIZE) \
		.batch(BATCH_SIZE)
	train_dataset = dataset.take(validation_batch_index).map(index_to_data)
	validation_dataset = dataset.skip(validation_batch_index).take(test_batch_index - validation_batch_index).map(
		index_to_data)
	test_dataset = dataset.skip(test_batch_index).map(index_to_data)

	return mean, std, train_dataset, validation_dataset, test_dataset

# with tf.Session() as sess:
# 	_, _, _, dataset, _ = get_datasets()
# 	get_next = dataset.make_one_shot_iterator().get_next()
# 	for i in range(10000):
# 		next = sess.run([get_next])
# 		print(i)

# train_dataset, validation_dataset, test_dataset = (
# 	Dataset.from_tensor_slices((seq_pair[0], seq_pair[1])).batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
# 	for seq_pair in
# 	zip(np.split(input_sequences, [validation_index, test_index]),
# 		np.split(target_sequences, [validation_index, test_index])))

# print(train_dataset)
