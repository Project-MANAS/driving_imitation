import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import nest

from manas.ai.planning.komanda.dataset.constant import *


class Komanda:
	def __init__(self, dataset_mean = 0, dataset_variance = 1):
		self.keep_prob = tf.placeholder_with_default(1.0, (), "keep_prob")
		self.aux_cost_weight = tf.placeholder_with_default(0.1, (), "aux_cost_weight")
		self.mean = dataset_mean
		self.var = dataset_variance
		self.info = {}
		self.output = {}
		self.TOWER_NAME = "KomandaBoi"

	@staticmethod
	def swish(x):
		return x * tf.nn.sigmoid(x)

	def apply_vision_simple(self, video, keep_prob, batch_size, seq_len, scope = None, reuse = None):
		with tf.variable_scope('vision', reuse=tf.AUTO_REUSE):
			with slim.arg_scope([slim.model_variable, slim.variable], device = '/cpu:0'):
				net = slim.convolution(video, 64, [3, 12, 12], [1, 6, 6], "VALID", activation_fn = self.swish)
				net = tf.nn.dropout(net, keep_prob)
				aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
				                            None)

				net = slim.convolution(net, 64, [2, 5, 5], [1, 2, 2], "VALID", activation_fn = self.swish)
				net = tf.nn.dropout(net, keep_prob)
				aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
				                            None)

				net = slim.convolution(net, 64, [2, 5, 5], [1, 1, 1], "VALID", activation_fn = self.swish)
				net = tf.nn.dropout(net, keep_prob)
				aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
				                            None)

				net = slim.convolution(net, 64, [2, 5, 5], [1, 1, 1], "VALID", activation_fn = self.swish)
				net = tf.nn.dropout(net, keep_prob)
				aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, None)

				net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, self.swish)
				net = tf.nn.dropout(net, keep_prob)
				net = slim.fully_connected(net, 512, self.swish)
				net = tf.nn.dropout(net, keep_prob)
				net = slim.fully_connected(net, 256, self.swish)
				net = tf.nn.dropout(net, keep_prob)
				net = slim.fully_connected(net, 128, None)
				x = self.swish(net + aux1 + aux2 + aux3 + aux4)

				return tf.contrib.layers.layer_norm(inputs = x, center = True, scale = True, activation_fn = None,
				                                    trainable = True)

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

		def __call__(self, inputs, state, scope = None):
			(visual_feats, current_ground_truth) = inputs
			prev_output, prev_state_internal = state
			context = tf.concat([prev_output, visual_feats], 1)
			new_output_internal, new_state_internal = self._internal_cell(context, prev_state_internal)
			new_output = tf.contrib.layers.fully_connected(
				inputs = tf.concat([new_output_internal, prev_output, visual_feats], 1),
				num_outputs = self._num_outputs,
				activation_fn = None,
				scope = "OutputProjection")
			# if self._use_ground_truth == True, we pass the ground truth as the state; otherwise, we use the model's predictions
			return new_output, (current_ground_truth if self._use_ground_truth else new_output, new_state_internal)

	graph = tf.get_default_graph()

	def build(self, scope, video, targets_normalized):
		with self.graph.as_default():
			with tf.name_scope(scope):
				visual_conditions_reshaped = self.apply_vision_simple(video, self.keep_prob, BATCH_SIZE, SEQ_LEN)

				visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
				visual_conditions = tf.nn.dropout(visual_conditions, self.keep_prob)

				rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
				rnn_inputs_autoregressive = (visual_conditions, tf.zeros((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), tf.float32))

				internal_cell = tf.nn.rnn_cell.LSTMCell(RNN_SIZE, num_proj = RNN_PROJ)
				cell_with_ground_truth = self.SamplingRNNCell(OUTPUT_DIM, True, internal_cell)
				cell_autoregressive = self.SamplingRNNCell(OUTPUT_DIM, False, internal_cell)

				def get_initial_state(complex_state_tuple_sizes):
					flat_sizes = nest.flatten(complex_state_tuple_sizes)
					init_state_flat = [tf.tile(
						multiples = [BATCH_SIZE, 1],
						input = tf.get_variable("controller_initial_state_%d" % i, initializer = tf.zeros_initializer,
						                        shape = ([1, s]),
						                        dtype = tf.float32))
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

				with tf.variable_scope("predictor_ground_truth"):
					out_gt, controller_final_state_gt = tf.nn.dynamic_rnn(
						cell = cell_with_ground_truth,
						inputs = rnn_inputs_with_ground_truth,
						sequence_length = [SEQ_LEN] * BATCH_SIZE,
						initial_state = controller_initial_state_gt,
						dtype = tf.float32,
						swap_memory = True,
						time_major = False)

				with tf.variable_scope("predictor_autoregressive"):
					out_autoregressive, controller_final_state_autoregressive = tf.nn.dynamic_rnn(
						cell = cell_autoregressive,
						inputs = rnn_inputs_autoregressive,
						sequence_length = [SEQ_LEN] * BATCH_SIZE,
						initial_state = controller_initial_state_autoregressive,
						dtype = tf.float32,
						swap_memory = True,
						time_major = False)

				with tf.name_scope("regression"):
					mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
					mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
					mse_autoregressive_steering = tf.reduce_mean(
						tf.squared_difference(out_autoregressive[:, :, 0], targets_normalized[:, :, 0]))
					steering_predictions = (out_autoregressive[:, :, 0] * self.var) + self.mean

					total_loss = mse_autoregressive_steering + self.aux_cost_weight * (mse_gt + mse_autoregressive)

					tf.add_to_collection('losses', total_loss)

					mse_autoregressive_steering_sqrt = tf.sqrt(mse_autoregressive_steering)
					mse_gt_sqrt = tf.sqrt(mse_gt)
					mse_autoregressive_sqrt = tf.sqrt(mse_autoregressive)

					self.output = {'mse_autoregressive_steering': mse_autoregressive_steering,
					               'controller_final_state_gt': controller_final_state_gt,
					               'controller_final_state_autoregressive': controller_final_state_autoregressive,
					               'controller_initial_state_autoregressive': controller_initial_state_autoregressive,
					               'steering_predictions': steering_predictions}

					self.info = {'mse_autoregressive_steering_sqrt': mse_autoregressive_steering_sqrt,
					             'mse_autoregressive_sqrt': mse_autoregressive_sqrt,
					             'mse_gt': mse_gt,
					             'mse_gt_sqrt': mse_gt_sqrt}