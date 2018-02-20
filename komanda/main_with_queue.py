import numpy as np
import tensorflow as tf

from manas.ai.planning.komanda import multi_gpu
from manas.ai.planning.komanda.dataset.constant import *
from manas.ai.planning.komanda.dataset.contrib_dataset import get_datasets
from manas.ai.planning.komanda.dataset.dataset import DatasetType
from manas.ai.planning.komanda.model import Komanda

iter_op, type_ops, mean, var = get_datasets()

lr = tf.placeholder_with_default(1e-3, (), tf.float32)
model = Komanda()
optimizer = tf.train.AdamOptimizer(lr)
train_op, summary_op, init_op = multi_gpu.train(model, optimizer, iter_op, 15.0)

mse_ar_steering = model.output['mse_autoregressive_steering']
final_state_gt = model.output['controller_final_state_gt']
final_state_ar = model.output['controller_final_state_autoregressive']
initial_state_ar = model.output['controller_initial_state_autoregressive']
steering_predictions = model.output['steering_predictions']

mse_ar_steering_sqrt = model.info['mse_autoregressive_steering_sqrt']
mse_ar_sqrt = model.info['mse_autoregressive_sqrt']
mse_gt = model.info['mse_ge']
mse_gt_sqrt = model.info['mse_gt_sqrt']

tf.summary.scalar('mse_autoregressive_steering_sqrt', mse_ar_steering_sqrt)
tf.summary.scalar('mse_gt_sqrt', mse_gt_sqrt)
tf.summary.scalar('mse_autoregressive_sqrt', mse_ar_sqrt)

summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/train_summary', graph = model.graph)
valid_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph = model.graph)
test_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph = model.graph)

saver = tf.train.Saver(write_version = tf.train.SaverDef.V2)

global_train_step = 0
global_valid_step = 0
global_test_step = 0

KEEP_PROB_TRAIN = 0.25
BATCH_COUNT = 12


def do_epoch(sess, type: DatasetType, batch_count):
	global global_train_step, global_valid_step, global_test_step

	state_gt_cur, state_ar_cur = None, None
	acc_loss = np.float128(0.0)

	if type == DatasetType.TRAIN:
		sess.run(type_ops['train'])
	elif type == DatasetType.VALIDATION:
		sess.run(type_ops['validation'])
	elif type == DatasetType.TEST:
		sess.run(type_ops['test'])

	for step in range(batch_count):
		feed_dict = {}
		ops_train = [summaries, train_op, mse_ar_steering, final_state_gt, final_state_ar]
		ops_valid = [steering_predictions, summaries, mse_ar_steering, final_state_ar]
		loss = 0.0

		if state_ar_cur is not None:
			feed_dict.update({initial_state_ar: state_ar_cur})
		if state_gt_cur is not None:
			feed_dict.update({final_state_gt: state_gt_cur})

		if type == DatasetType.TRAIN:
			feed_dict.update({model.keep_prob: KEEP_PROB_TRAIN})
			summary, _, loss, state_gt_cur, state_ar_cur = session.run(ops_train, feed_dict = feed_dict)
			train_writer.add_summary(summary, global_train_step)
			global_train_step += 1

		elif type == DatasetType.VALIDATION:
			model_predictions, summary, loss, state_ar_cur = session.run(ops_valid, feed_dict = feed_dict)
			valid_writer.add_summary(summary, global_valid_step)
			global_valid_step += 1

		elif type == DatasetType.TEST:
			model_predictions, summary, loss, state_ar_cur = session.run(ops_valid, feed_dict = feed_dict)
			test_writer.add_summary(summary, global_test_step)
			global_test_step += 1

		acc_loss += loss
		print('Iter', step, '\t', loss)

	avg_loss = acc_loss / batch_count
	print('Average loss this epoch:', avg_loss)

	return avg_loss


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95, allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options, log_device_place = True, allow_soft_placement = True)
best_validation_score = None

with tf.Session(graph = model.graph, config = config) as session:
	session.run(init_op)
	print('Initialized')

	ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
	if ckpt and False:
		print("Restoring from", ckpt)
		saver.restore(sess = session, save_path = ckpt)

	for epoch in range(NUM_EPOCHS):
		print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

		print("Validation:")
		valid_score = do_epoch(session, DatasetType.VALIDATION, 12)

		if best_validation_score is None:
			best_validation_score = valid_score
		if valid_score < best_validation_score:
			saver.save(session, CHECKPOINT_DIR + '/checkpoint-sdc-ch2')
			best_validation_score = valid_score
			print('\r', "SAVED at epoch %d" % epoch)

		if epoch != NUM_EPOCHS - 1:
			print("Training")
			do_epoch(session, DatasetType.TRAIN, 12)

		if epoch == NUM_EPOCHS - 1:
			print("Test:")
			test_score = do_epoch(session, DatasetType.TEST, 12)
			print('Final test loss:', test_score)
