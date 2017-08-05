import os

import numpy as np

from manas.ai.planning.komanda.dataset import dataset

# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT. 
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left, decreasing the sequence length.
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

CHECKPOINT_DIR = os.environ['CHECKPOINTS'] + "/udacity_steering/challenge_2/v3"
DATASET_DIR = os.environ['DATASETS'] + "/udacity_steering/challenge_2"

train_seq, valid_seq, test_seq, mean, std = dataset.process_csv("interpolated.csv", DATASET_DIR + "/bag_extraction",
																OUTPUT_DIM,
																SEQ_LEN,
																BATCH_SIZE,
																val=5,
																test=2)  # concatenated interpolated.csv from rosbags

import tensorflow as tf
from tensorflow.python.util import nest

slim = tf.contrib.slim

with tf.device('/gpu:0'):
	layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None,
														trainable=True)


def get_optimizer(loss, lrate):
	optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
	gradvars = optimizer.compute_gradients(loss)
	gradients, v = zip(*gradvars)
	gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
	return optimizer.apply_gradients(zip(gradients, v))


def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
	video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])
	with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):
		net = slim.convolution(video, num_outputs=64, kernel_size=[3, 12, 12], stride=[1, 6, 6], padding="VALID")
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
									activation_fn=None)

		net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 2, 2], padding="VALID")
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
									activation_fn=None)

		net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
									activation_fn=None)

		net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		# at this point the tensor 'net' is of shape batch_size x seq_len x ...
		aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)

		net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
		net = tf.nn.dropout(x=net, keep_prob=keep_prob)
		net = slim.fully_connected(net, 128, activation_fn=None)
		return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4))  # aux[1-4] are residual connections (shortcuts)


class SamplingRNNCell(tf.nn.rnn_cell.RNNCell):
	"""Simple sampling RNN cell."""

	def __init__(self, num_outputs, use_ground_truth, internal_cell):
		"""
		if use_ground_truth then don't sample
		"""
		self._num_outputs = num_outputs
		self._use_ground_truth = use_ground_truth  # boolean
		self._internal_cell = internal_cell  # may be LSTM or GRU or anything

	@property
	def state_size(self):
		return self._num_outputs, self._internal_cell.state_size  # previous output and bottleneck state

	@property
	def output_size(self):
		return self._num_outputs  # steering angle, torque, vehicle speed

	def __call__(self, inputs, state, scope=None):
		(visual_feats, current_ground_truth) = inputs
		prev_output, prev_state_internal = state
		context = tf.concat([prev_output, visual_feats], 1)
		new_output_internal, new_state_internal = internal_cell(context,
																prev_state_internal)  # here the internal cell (e.g. LSTM) is called
		new_output = tf.contrib.layers.fully_connected(
			inputs=tf.concat([new_output_internal, prev_output, visual_feats], 1),
			num_outputs=self._num_outputs,
			activation_fn=None,
			scope="OutputProjection")
		# if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
		return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)


graph = tf.Graph()

with graph.as_default():
	# inputs
	learning_rate = tf.placeholder_with_default(input=1e-4, shape=())
	keep_prob = tf.placeholder_with_default(input=1.0, shape=())
	aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())

	inputs = tf.placeholder(shape=(BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN),
							dtype=tf.string)  # pathes to png files from the central camera
	targets = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
							 dtype=tf.float32)  # seq_len x batch_size x OUTPUT_DIM

	input_images = tf.stack([tf.image.decode_png(tf.read_file(x)) for x in
							 tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE]))])

	with tf.device('/gpu:0'):
		targets_normalized = (targets - mean) / std

		input_images = tf.cast(input_images, tf.float32) * 2.0 / 255.0 - 1.0
		input_images.set_shape([(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
		visual_conditions_reshaped = apply_vision_simple(image=input_images, keep_prob=keep_prob,
														 batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
		visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
		visual_conditions = tf.nn.dropout(x=visual_conditions, keep_prob=keep_prob)

		rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
		rnn_inputs_autoregressive = (
			visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32))

		internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
		cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True,
												 internal_cell=internal_cell)
		cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False,
											  internal_cell=internal_cell)


	def get_initial_state(complex_state_tuple_sizes):
		flat_sizes = nest.flatten(complex_state_tuple_sizes)
		init_state_flat = [tf.tile(
			multiples=[BATCH_SIZE, 1],
			input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer, shape=([1, s]),
								  dtype=tf.float32))
			for i, s in enumerate(flat_sizes)]
		init_state = nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
		return init_state


	def deep_copy_initial_state(complex_state_tuple):
		flat_state = nest.flatten(complex_state_tuple)
		flat_copy = [tf.identity(s) for s in flat_state]
		deep_copy = nest.pack_sequence_as(complex_state_tuple, flat_copy)
		return deep_copy


	with tf.device('/gpu:0'):
		controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
		controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
		controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)

		with tf.variable_scope("predictor"):
			out_gt, controller_final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth,
																  inputs=rnn_inputs_with_ground_truth,
																  sequence_length=[SEQ_LEN] * BATCH_SIZE,
																  initial_state=controller_initial_state_gt,
																  dtype=tf.float32,
																  swap_memory=True, time_major=False)
		with tf.variable_scope("predictor", reuse=True):
			out_autoregressive, controller_final_state_autoregressive = tf.nn.dynamic_rnn(cell=cell_autoregressive,
																						  inputs=rnn_inputs_autoregressive,
																						  sequence_length=[
																											  SEQ_LEN] * BATCH_SIZE,
																						  initial_state=controller_initial_state_autoregressive,
																						  dtype=tf.float32,
																						  swap_memory=True,
																						  time_major=False)
		mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
		mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
		mse_autoregressive_steering = tf.reduce_mean(
			tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
		steering_predictions = (out_autoregressive[:, :, 0] * std[0]) + mean[0]

		total_loss = mse_autoregressive_steering + aux_cost_weight * (mse_gt + mse_autoregressive)

		optimizer = get_optimizer(total_loss, learning_rate)

		mse_autoregressive_steering_sqrt = tf.sqrt(mse_autoregressive_steering)
		mse_gt_sqrt = tf.sqrt(mse_gt)
		mse_autoregressive_sqrt = tf.sqrt(mse_autoregressive)

		tf.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", mse_autoregressive_steering_sqrt)
		tf.summary.scalar("rmse_gt", mse_gt_sqrt)
		tf.summary.scalar("rmse_autoregressive", mse_autoregressive_sqrt)

		summaries = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/train_summary', graph=graph)
		valid_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph=graph)
		test_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph=graph)
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

global_train_step = 0
global_valid_step = 0
global_test_step = 0

KEEP_PROB_TRAIN = 0.25


def do_epoch(session, sequences, mode):
	global global_train_step, global_valid_step, test_valid_step
	test_predictions = {}
	valid_predictions = {}
	batch_generator = dataset.BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
											 left_context=LEFT_CONTEXT)
	total_num_steps = int(1 + (batch_generator.indices[1] - 1) / SEQ_LEN)
	controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
	acc_loss = np.float128(0.0)
	for step in range(total_num_steps):
		feed_inputs, feed_targets = batch_generator.next()
		feed_dict = {inputs: feed_inputs, targets: feed_targets}
		if controller_final_state_autoregressive_cur is not None:
			feed_dict.update({controller_initial_state_autoregressive: controller_final_state_autoregressive_cur})
		if controller_final_state_gt_cur is not None:
			feed_dict.update({controller_final_state_gt: controller_final_state_gt_cur})
		if mode == "train":
			feed_dict.update({keep_prob: KEEP_PROB_TRAIN})
			summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
				session.run([summaries, optimizer, mse_autoregressive_steering, controller_final_state_gt,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			train_writer.add_summary(summary, global_train_step)
			global_train_step += 1
		elif mode == "valid":
			model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
				session.run([steering_predictions, summaries, mse_autoregressive_steering,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			valid_writer.add_summary(summary, global_valid_step)
			global_valid_step += 1
			steering_targets = feed_targets[:, :, 0].flatten()
			feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
			model_predictions = model_predictions.flatten()
			stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
			for i, img in enumerate(feed_inputs):
				valid_predictions[img] = stats[:, i]
		elif mode == "test":
			model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
				session.run([steering_predictions, summaries, mse_autoregressive_steering,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			test_writer.add_summary(summary, global_test_step)
			feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
			model_predictions = model_predictions.flatten()
			for i, img in enumerate(feed_inputs):
				test_predictions[img] = model_predictions[i]
		acc_loss += loss
		print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step + 1)), )
	return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)


NUM_EPOCHS = 100

best_validation_score = None
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options,
												   # log_device_placement=True,
												   allow_soft_placement=True)) as session:
	with tf.device('/gpu:0'):
		session.run(tf.global_variables_initializer())
		print('Initialized')
		ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
		if ckpt:
			print("Restoring from", ckpt)
			saver.restore(sess=session, save_path=ckpt)
		for epoch in range(NUM_EPOCHS):
			print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

			print("Validation:")
			valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")

			if best_validation_score is None:
				best_validation_score = valid_score
			if valid_score < best_validation_score:
				saver.save(session, CHECKPOINT_DIR + '/checkpoint-sdc-ch2')
				best_validation_score = valid_score
				print('\r', "SAVED at epoch %d" % epoch, )
				with open(CHECKPOINT_DIR + "/valid-predictions-epoch%d" % epoch, "w") as out:
					result = np.float128(0.0)
					for img, stats in valid_predictions.items():
						print(img, stats) >> out
						result += stats[-1]
				print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)))

			if epoch != NUM_EPOCHS - 1:
				print("Training")
				do_epoch(session=session, sequences=train_seq, mode="train")

			if epoch == NUM_EPOCHS - 1:
				print("Test:")
				with open(CHECKPOINT_DIR + "/test-predictions-epoch%d" % epoch, "w") as out:
					_, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
					print("frame_id,steering_angle")
					for img, pred in test_predictions.items():
						img = img.replace("challenge_2/Test-final/center/", "")
						print("%s,%f" % (img, pred))
