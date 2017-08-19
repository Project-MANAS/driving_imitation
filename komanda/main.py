import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import manas.ai.planning.komanda.dataset.contrib_dataset as contrib_dataset
from manas.ai.planning.komanda.dataset.constant import *
from manas.ai.planning.komanda.dataset.dataset import DatasetType
from manas.ai.planning.komanda.dataset.pipeline import Pipeline, PipelineOptions

slim = tf.contrib.slim

layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True, activation_fn=None,
													trainable=True)


def get_optimizer(loss, lrate):
	with tf.name_scope("optimizer"):
		optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
		gradvars = optimizer.compute_gradients(loss)
		gradients, v = zip(*gradvars)
		gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
		return optimizer.apply_gradients(zip(gradients, v))


def apply_vision_simple(video, keep_prob, batch_size, seq_len, scope=None, reuse=None):
	with tf.variable_scope(scope, 'Vision', [video], reuse=reuse):
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
		new_output_internal, new_state_internal = self._internal_cell(context,
																	  prev_state_internal)  # here the internal cell (e.g. LSTM) is called
		new_output = tf.contrib.layers.fully_connected(
			inputs=tf.concat([new_output_internal, prev_output, visual_feats], 1),
			num_outputs=self._num_outputs,
			activation_fn=None,
			scope="OutputProjection")
		# if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
		return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)


graph = tf.get_default_graph()

with graph.as_default():
	with tf.name_scope("model"):
		# inputs
		learning_rate = tf.placeholder_with_default(input=1e-4, shape=(), name="learning_rate")
		keep_prob = tf.placeholder_with_default(input=1.0, shape=(), name="keep_prob")
		aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=(), name="aux_cost_weight")

		targets_normalized = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32)
		video = tf.placeholder(shape=[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS],
							   dtype=tf.float32,
							   name="video_input")
		visual_conditions_reshaped = apply_vision_simple(video=video, keep_prob=keep_prob,
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
				input=tf.get_variable("controller_initial_state_%d" % i, initializer=tf.zeros_initializer,
									  shape=([1, s]),
									  dtype=tf.float32))
				for i, s in enumerate(flat_sizes)]
			init_state = nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
			return init_state


		def deep_copy_initial_state(complex_state_tuple):
			flat_state = nest.flatten(complex_state_tuple)
			flat_copy = [tf.identity(s) for s in flat_state]
			deep_copy = nest.pack_sequence_as(complex_state_tuple, flat_copy)
			return deep_copy


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

		with tf.name_scope("regression"):
			mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
			mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
			mse_autoregressive_steering = tf.reduce_mean(
				tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
			steering_predictions = (out_autoregressive[:, :, 0] * contrib_dataset.std[0]) + contrib_dataset.mean[0]

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

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)

global_train_step = 0
global_valid_step = 0
global_test_step = 0

KEEP_PROB_TRAIN = 0.25


def do_epoch(sess, pipeline, type: DatasetType):
	global global_train_step, global_valid_step, test_valid_step, test_set
	test_predictions = {}
	valid_predictions = {}

	# TODO Stop the enqueue threads and flush the pipeline here before restarting the enqueue threads with the new dataset before each epoch
	# pipeline = pipelines[type.value]
	batch_container = batch_containers[type.value]
	# dataset = batch_container.dataset
	batch_count = batch_container.count

	# data_batch_op, target_batch_op = pipeline.get_batches()
	controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
	acc_loss = np.float128(0.0)

	# TODO Find more elegant way to get size of dataset (batch_containers should be abstracted away since only the pipeline should interact with it directly)
	# TODO Actual queue would be filled with less than batch_count * N_AUG if one of the threads do not augment (Solved by issue above)
	for step in range(batch_count * N_AUG):
		print("Request dequeu")
		batch = sess.run(pipeline.dequeue_op)
		print("Deque satisfied")
		feed_dict = {video: batch[0], targets_normalized: batch[1]}
		if controller_final_state_autoregressive_cur is not None:
			feed_dict.update({controller_initial_state_autoregressive: controller_final_state_autoregressive_cur})
		if controller_final_state_gt_cur is not None:
			feed_dict.update({controller_final_state_gt: controller_final_state_gt_cur})
		if type == DatasetType.TRAIN:
			feed_dict.update({keep_prob: KEEP_PROB_TRAIN})
			# TODO Find way to increase GPU utilization by optimizing network
			summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
				session.run([summaries, optimizer, mse_autoregressive_steering, controller_final_state_gt,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			train_writer.add_summary(summary, global_train_step)
			global_train_step += 1
		elif type == DatasetType.VALIDATION:
			model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
				session.run([steering_predictions, summaries, mse_autoregressive_steering,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			valid_writer.add_summary(summary, global_valid_step)
			global_valid_step += 1
		# TODO Fix:
		# steering_targets = curr_target_batch[:, :, 0].flatten()
		# feed_inputs = curr_input_batch[:, LEFT_CONTEXT:].flatten()
		# model_predictions = model_predictions.flatten()
		# stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
		# for i, img in enumerate(feed_inputs):
		# 	valid_predictions[img] = stats[:, i]
		elif type == DatasetType.TEST:
			model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
				session.run([steering_predictions, summaries, mse_autoregressive_steering,
							 controller_final_state_autoregressive],
							feed_dict=feed_dict)
			test_writer.add_summary(summary, global_test_step)
		# TODO Print predictions along side image paths by including image paths in pipeline
		# feed_inputs = get_next[:, LEFT_CONTEXT:].flatten()
		# model_predictions = model_predictions.flatten()
		# for i, img in enumerate(feed_inputs):
		# 	test_predictions[img] = model_predictions[i]
		else:
			raise AttributeError('Unknown dataset type')
		acc_loss += loss
		print('\r', step + 1, "/", batch_count * N_AUG, np.sqrt(acc_loss / (step + 1)), )

	return (np.sqrt(acc_loss / batch_count), valid_predictions) if type != DatasetType.TEST \
		else (None, test_predictions)


NUM_EPOCHS = 100

best_validation_score = None
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options,
												   # log_device_placement=True,
												   allow_soft_placement=True)) as session:
	session.run(tf.global_variables_initializer())

	print('Initialized')
	ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
	if ckpt and False:
		print("Restoring from", ckpt)
		saver.restore(sess=session, save_path=ckpt)

	with tf.Session() as sess:
		with tf.name_scope("pipeline"):
			mean, std, batch_containers = contrib_dataset.get_datasets()
			pipelineOptions = PipelineOptions()
			# TODO Batch container should be set instead of passed to constructor to allow dataset to be changed at runtime
			pipeline = Pipeline(graph, sess, batch_containers[DatasetType.TRAIN.value], pipelineOptions)
			# TODO Pipeline only needs to be started within do_epoch
			pipeline.start_pipeline()

		for epoch in range(NUM_EPOCHS):
			print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

			print("Validation:")
			valid_score, valid_predictions = do_epoch(sess=sess, pipeline=pipeline,
													  type=DatasetType.VALIDATION)

			if best_validation_score is None:
				best_validation_score = valid_score
			if valid_score < best_validation_score:
				saver.save(session, CHECKPOINT_DIR + '/checkpoint-sdc-ch2')
				best_validation_score = valid_score
				print('\r', "SAVED at epoch %d" % epoch, )
				with open(CHECKPOINT_DIR + "/valid-predictions-epoch%d" % epoch, "w") as out:
					result = np.float128(0.0)
					for img, stats in valid_predictions.items():
						print(img, stats)
						result += stats[-1]
				print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)))

			if epoch != NUM_EPOCHS - 1:
				print("Training")
				do_epoch(sess=sess, pipeline=pipeline, type=DatasetType.TRAIN)

			if epoch == NUM_EPOCHS - 1:
				print("Test:")
				with open(CHECKPOINT_DIR + "/test-predictions-epoch%d" % epoch, "w") as out:
					_, test_predictions = do_epoch(sess=sess, pipeline=pipeline, type=DatasetType.TEST)
					print("frame_id,steering_angle")
					for img, pred in test_predictions.items():
						img = img.replace("challenge_2/Test-final/center/", "")
						print("%s,%f" % (img, pred))
