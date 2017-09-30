import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from manas.ai.dataset.queue import HomoThreadHeteroQueue, TfFIFOQueue, HeteroThreadHeteroQueue
from manas.ai.planning.komanda import model
from manas.ai.planning.komanda.dataset import contrib_dataset
from manas.ai.planning.komanda.dataset.constant import *
from manas.ai.planning.komanda.dataset.contrib_dataset import BatchContainers
from manas.ai.planning.komanda.dataset.dataset import DatasetType

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


def on_build_callback(thread_index, mqueue, placeholders, sess, batch_containers: BatchContainers):
	enqueue_input = mqueue.queues[0].enqueue(placeholders)
	return mqueue, enqueue_input, sess, batch_containers


def on_start_callback(thread_index, thread_handler: HeteroThreadHeteroQueue.ThreadHandler,
					  mqueue: HomoThreadHeteroQueue, enqueue_input, sess,
					  batch_containers: BatchContainers):
	batch_container = batch_containers.curr_batch_container
	with sess.as_default():
		sess.run(batch_container.initialize_iterator)
		while not mqueue.coord.should_stop():
			try:
				batch = sess.run(batch_container.get_next)
				print("Batch fetched %d" % thread_index)
				videos, targets = batch[0], batch[1]
				sess.run(enqueue_input, feed_dict={video: videos, targets_normalized: targets})
			except OutOfRangeError:
				print('Thread %d completed' % thread_index)
				return


def build_datapipe(capacity, sess, batch_containers: BatchContainers):
	input_queue = HomoThreadHeteroQueue([], 0, sess,
										(on_build_callback, on_start_callback),
										None, (video, targets_normalized))
	input_queue.add_queues([
		TfFIFOQueue(
			tf.FIFOQueue(capacity=capacity, shapes=[[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS],
													[BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]],
						 dtypes=[tf.float32, tf.float32]))  # Video

	])
	# input_queue.add_queues([
	# 	StagingAreaQueue(
	# 		tf.contrib.staging.StagingArea(capacity=capacity,shapes=[[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS]],
	# 									   dtypes=[tf.float32])),
	# 	# Video
	# 	StagingAreaQueue(
	# 		tf.contrib.staging.StagingArea(capacity=capacity,shapes=[[BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]],
	# 									   dtypes=[tf.float32]))  # Normalized targets
	# ])
	placeholders = (video, targets_normalized)
	batch_containers.build_graph()
	input_queue.add_threads_from_callback((on_build_callback, on_start_callback),
										  4, input_queue,
										  placeholders, sess, batch_containers)

	# input_queue.add_thread_from_callback((on_build_callback, on_start_callback),input_queue,(video,targets_normalized),sess,batch_container)
	input_queue.set_session(sess)
	return input_queue


def do_epoch(sess, batch_containers: BatchContainers, type: DatasetType):
	global global_train_step, global_valid_step, test_valid_step, test_set
	valid_predictions = {}
	test_predictions = {}

	batch_containers.curr_type = type.value
	input_queue.start()

	batch_container = batch_containers.curr_batch_container
	batch_count = batch_container.count
	controller_final_state_gt_cur, controller_final_state_autoregressive_cur = None, None
	acc_loss = np.float128(0.0)
	get_batch_op = input_queue.queues[0].dequeue()
	for step in range(batch_count):
		print("Requesting Dequeue")
		batch = sess.run(get_batch_op)
		print("Dequeued successfully")
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
	input_queue.stop_all(queue_stop_mode='clear')
	return (np.sqrt(acc_loss / batch_count), valid_predictions) if type != DatasetType.TEST else (
		None, test_predictions)


NUM_EPOCHS = 100
QUEUE_CAPACITY = 100
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)

best_validation_score = None
with tf.Session(graph=model.graph, config=tf.ConfigProto(gpu_options=gpu_options,
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
		with tf.name_scope("datapipe"):
			input_queue = build_datapipe(QUEUE_CAPACITY, sess, batch_containers)
		# TODO Batch container should be set instead of passed to constructor to allow dataset to be changed at runtime

		for epoch in range(NUM_EPOCHS):
			print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

			print("Validation:")
			valid_score, valid_predictions = do_epoch(sess=sess, batch_containers=batch_containers,
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
				do_epoch(sess=sess, batch_containers=batch_containers, type=DatasetType.TRAIN)

			if epoch == NUM_EPOCHS - 1:
				print("Test:")
				with open(CHECKPOINT_DIR + "/test-predictions-epoch%d" % epoch, "w") as out:
					_, test_predictions = do_epoch(sess=sess, batch_containers=batch_containers, type=DatasetType.TEST)
					print("frame_id,steering_angle")
					for img, pred in test_predictions.items():
						img = img.replace("challenge_2/Test-final/center/", "")
						print("%s,%f" % (img, pred))
