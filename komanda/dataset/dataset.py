import os
from enum import Enum

import numpy as np

# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT.
# Batch size specifies the number of sequence fragments used in a single optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left,
# decreasing the sequence length.
# One should be careful here to maintain the model's causality.
SEQ_LEN = 10
BATCH_SIZE = 5
LEFT_CONTEXT = 5

# These are the input image parameters.
HEIGHT = 480
WIDTH = 640
CHANNELS = 3  # RGB

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3]  # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS)  # predict all features: steering angle, torque and vehicle speed

DATASET_DIR = os.environ['DATASETS'] + "/udacity_steering/challenge_2"


class BatchGenerator(object):
	def __init__(self, sequence, seq_len, batch_size, left_context):
		self.sequence = sequence
		self.seq_len = seq_len
		self.batch_size = batch_size
		chunk_size = int(1 + (len(sequence) - 1) / batch_size)
		self.indices = [(i * chunk_size) % len(sequence) for i in range(batch_size)]
		self.left_context = left_context

	def next(self):
		inputs_batch = []
		targets_batch = []
		output = []
		for i in range(self.batch_size):
			idx = self.indices[i]
			left_pad = self.sequence[idx - self.left_context:idx]
			if len(left_pad) < self.left_context:
				left_pad = [self.sequence[0]] * (self.left_context - len(left_pad)) + left_pad
			assert len(left_pad) == self.left_context
			leftover = len(self.sequence) - idx
			if leftover >= self.seq_len:
				result = self.sequence[idx:idx + self.seq_len]
			else:
				result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
			assert len(result) == self.seq_len
			self.indices[i] = (idx + self.seq_len) % len(self.sequence)
			images, targets = zip(*result)
			images_left_pad, _ = zip(*left_pad)
			inputs_batch.append(np.stack(images_left_pad + images))
			targets_batch.append(targets)
			output.append((np.stack(images_left_pad + images), np.stack(targets)))
		output = list(zip(*output))
		output[0] = np.stack(output[0])  # batch_size x (self.left_context + seq_len)
		output[1] = np.stack(output[1])  # batch_size x seq_len x OUTPUT_DIM
		# return output
		return [np.stack(inputs_batch), np.stack(targets_batch)]


class DatasetType(Enum):
	TRAIN = 0
	VALIDATION = 1
	TEST = 2


class DatasetIndices(object):
	DATASET_DICT = None
	TRAIN = None
	VALIDATION = None
	TEST = None

	def __init__(self, seq, mean, std):
		self.sequence = seq
		self.mean = mean
		self.std = std

	@staticmethod
	def read_csv(filename, dataset_dir):
		with open(filename, 'r') as f:
			f.readline()
			lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
			lines = map(lambda x: (dataset_dir + "/" + x[0], np.float32(x[1:])), lines)  # imagefile, outputs
			return lines

	@staticmethod
	def load(filename, dataset_dir, output_dim, seq_len, batch_size, val=5, test=2):
		# Might as well read entire dataset since mean and std has to be calculated for entire dataset
		sum_f = np.float128([0.0] * output_dim)
		sum_sq_f = np.float128([0.0] * output_dim)
		lines = DatasetIndices.read_csv(dataset_dir + "/" + filename, dataset_dir)
		# leave val% for validation and test% for test
		train_seq = []
		valid_seq = []
		test_seq = []
		cnt = 0
		frames_per_batch = seq_len * batch_size
		for ln in lines:
			if cnt < frames_per_batch * (100 - val - test):
				train_seq.append(ln)
				sum_f += ln[1]
				sum_sq_f += ln[1] * ln[1]
			elif cnt < frames_per_batch * (100 - test):
				valid_seq.append(ln)
			else:
				test_seq.append(ln)
			cnt += 1
			cnt %= frames_per_batch * 100
			mean = sum_f / len(train_seq)
		var = sum_sq_f / len(train_seq) - mean ** 2
		std = np.sqrt(var)
		print({'train_seq': len(train_seq), 'valid_seq': len(valid_seq)})
		# we will need these statistics to normalize the outputs (and ground truth inputs)
		print({'mean': mean, 'std': std})
		DatasetIndices.TRAIN = DatasetIndices(train_seq, mean, std)
		DatasetIndices.VALIDATION = DatasetIndices(valid_seq, mean, std)
		DatasetIndices.TEST = DatasetIndices(test_seq, mean, std)
		DatasetIndices.DATASET_DICT = {
			DatasetType.TRAIN: DatasetIndices.TRAIN,
			DatasetType.VALIDATION: DatasetIndices.VALIDATION,
			DatasetType.TEST: DatasetIndices.TEST
		}

	@staticmethod
	def load_default():
		DatasetIndices.load("interpolated.csv",
							DATASET_DIR + "/bag_extraction",
							OUTPUT_DIM,
							SEQ_LEN,
							BATCH_SIZE,
							val=5,
							test=2)  # concatenated interpolated.csv from rosbags


# TODO Remove
def read_csv(filename, dataset_dir):
	with open(filename, 'r') as f:
		f.readline()
		lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
		lines = map(lambda x: (dataset_dir + "/" + x[0], np.float32(x[1:])), lines)  # imagefile, outputs
		return lines


# TODO Remove
def process_csv(filename, dataset_dir, output_dim, seq_len, batch_size, val=5, test=2):
	sum_f = np.float128([0.0] * output_dim)
	sum_sq_f = np.float128([0.0] * output_dim)
	lines = read_csv(dataset_dir + "/" + filename, dataset_dir)
	# leave val% for validation
	train_seq = []
	valid_seq = []
	test_seq = []
	cnt = 0
	frames_per_batch = seq_len * batch_size
	for ln in lines:
		if cnt < frames_per_batch * (100 - val - test):
			train_seq.append(ln)
			sum_f += ln[1]
			sum_sq_f += ln[1] * ln[1]
		elif cnt < frames_per_batch * (100 - test):
			valid_seq.append(ln)
		else:
			test_seq.append(ln)
		cnt += 1
		cnt %= frames_per_batch * 100
	mean = sum_f / len(train_seq)
	var = sum_sq_f / len(train_seq) - mean * mean
	std = np.sqrt(var)
	print({'train_seq': len(train_seq), 'valid_seq': len(valid_seq)})
	print(
		{'mean': mean, 'std': std})  # we will need these statistics to normalize the outputs (and ground truth inputs)
	return train_seq, valid_seq, test_seq, mean, std
