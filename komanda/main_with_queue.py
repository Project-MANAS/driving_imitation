import tensorflow as tf

from manas.ai.dataset import queue
from manas.ai.planning.komanda import model
from manas.ai.planning.komanda.dataset.constant import *

model_input, model_output, model_info, optimizer = model.get_model_params()  # Tf komanada model graph output operations
targets_normalized, video, keep_prob = model_input
mse_autoregressive_steering, controller_final_state_gt, controller_final_state_autoregressive, controller_initial_state_autoregressive, steering_predictions = model_output
mse_autoregressive_steering_sqrt, mse_autoregressive_sqrt, mse_gt, mse_gt_sqrt = model_info

tf.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", mse_autoregressive_steering_sqrt)
tf.summary.scalar("rmse_gt", mse_gt_sqrt)
tf.summary.scalar("rmse_autoregressive", mse_autoregressive_sqrt)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/train_summary', graph=model.graph)
valid_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph=model.graph)
test_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph=model.graph)
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

global_train_step = 0
global_valid_step = 0
global_test_step = 0

KEEP_PROB_TRAIN = 0.25


def on_build_callback(thread_index, mqueue, placeholders):
	enqueue_many = mqueue.enqueue_many(placeholders)
	return enqueue_many


def on_start_callback(thread_index, enqueue_many, mqueue, sess, batch_container):
	_next_op = batch_container.dataset.make_one_shot_iterator().get_next()
	with sess.as_default():
		batch = sess.run(_next_op)
		print('Enqueue thread %d: Batch fetched' % thread_index)
		enqueue_many(batch)


def initializeQueue(capacity):
	input_queue = queue.HeteroThreadMonoQueue()
	input_queue.add_queue(tf.FIFOQueue(shape=[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS],
									   dtype=tf.float32), name="video")  # Video
	input_queue.add_queue(
		tf.FIFOQueue(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32, name="targets"))  # Normalized targets

# input_queue.add_threads_from_callback((on_build_default_graph, on_default_start))
