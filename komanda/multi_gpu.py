import re

import tensorflow as tf

NUM_GPUS = 2
MOVING_AVERAGE_DECAY = 0.9999

"""
How to build a model to take advantage of multi-GPUs:
1. Code has to written in low-level Tensorflow using tf.nn or tf.rnn
2. tf.Variable declarations must be done via _variable_with_weight_decay
3. Loses need to be added to the collection `losses` after being appropriately weighed down
4. Should contain a build method that creates the forward pass of the graph with a tf.data.Iterator
5. Models need a parameter called `TOWER_NAME` 
"""


def variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable
	Returns:
	  Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer = initializer, dtype = tf.float32)
	return var


def variable_with_weight_decay(name, shape, lambda_reg = None):
	"""Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with Xavier/Glorot initialization.
	A weight decay is added only if one is specified.
	Args:
	  name: name of the variable
	  shape: list of ints
	  lambda_reg: add L2Loss weight decay multiplied by this float. If None, weight
		  decay is not added for this Variable.
	Returns:
	  Variable Tensor
	"""
	var = variable_on_cpu(name, shape, tf.glorot_normal_initializer())
	if lambda_reg is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), lambda_reg, name = 'weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def tower_loss(scope, model, iter_op):
	"""Calculate the total loss on a single tower

	Args:
	    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
	    model: user definied Tensorflow network with a `build` method
	    iter_op: Tensorflow.data.Iterator from your data.Dataset

	Returns:
	    Scalar op containing the total loss for a batch of data
	"""

	model.build(scope, *iter_op)
	losses = tf.get_collection('losses', scope)
	total_loss = tf.add_n(losses, name = 'total_loss')

	for l in losses + [total_loss]:
		loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
		tf.summary.scalar(loss_name, l)

	return total_loss


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
	  tower_grads: List of lists of (gradient, variable) tuples. The outer list
		is over individual gradients. The inner list is over the gradient
		calculation for each tower.
	Returns:
	   List of pairs of (gradient, variable) where the gradient has been averaged
	   across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis = 0, values = grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def train(model, optimizer, iter_op, clip_param = None):
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		# Create a variable to count the number of train() calls. This equals the
		# number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable(
			'global_step', [],
			initializer = tf.constant_initializer(0), trainable = False)

		tower_grads = []
		summaries = None
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(NUM_GPUS):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
						loss = tower_loss(scope, model, iter_op)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						gradvars = optimizer.compute_gradients(loss, colocate_gradients_with_ops = True)

						if clip_param is not None:
							gradients, v = zip(*gradvars)
							gradients, _ = tf.clip_by_global_norm(gradients, clip_param)
							gradvars = zip(gradients, v)

						tower_grads.append(gradvars)

		# We must calculate the mean of each gradient. Note that this is the
		# synchronization point across all towers.
		grads = average_gradients(tower_grads)

		# Add a summary to track the learning rate.
		if hasattr(optimizer, '_lr'):
			summaries.append(tf.summary.scalar('learning_rate', optimizer._lr))
		elif hasattr(optimizer, '_learning_rate'):
			summaries.append(tf.summary.scalar('learning_rate', optimizer._learning_rate))
		else:
			raise AttributeError("Optimzer does not have learning rate attribute")

		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

		# Apply the gradients to adjust the shared variables.
		apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)

		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram(var.op.name, var))

		# Track the moving averages of all trainable variables.
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		# Group all updates to into a single train op.
		train_op = tf.group(apply_gradient_op, variables_averages_op)

		# Build the summary operation from the last tower summaries.
		summary_op = tf.summary.merge(summaries)

		# Build an initialization operation to run below.
		init = tf.global_variables_initializer()

		# Start running operations on the Graph. allow_soft_placement must be set to
		# True to build towers on GPU, as some of the ops do not have GPU
		# implementations.

		return train_op, summary_op, init
