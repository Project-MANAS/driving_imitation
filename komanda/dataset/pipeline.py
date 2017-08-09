import tensorflow as tf
import threading

from manas.ai.dataset.input_pipeline import InputPipeline, InputPipelineOptions
from manas.ai.planning.komanda.dataset.dataset import *

pipelineOptions = InputPipelineOptions()
pipelineOptions.batch_size = 5
pipelineOptions.enqueue_size = 1
pipelineOptions.min_after_dequeue = 6

class Pipeline(object):
	def __init__(self, graph, sess, options: InputPipelineOptions):
		self.graph = graph
		self.sess = sess
		self.dataset_indices = None

		data_shape = [BATCH_SIZE, (LEFT_CONTEXT + SEQ_LEN), HEIGHT, WIDTH, CHANNELS]
		data_dtype = tf.float32
		target_shape = [BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]
		target_dtype = tf.float32

		self.input_placeholder = tf.placeholder(dtype=data_dtype, shape=data_shape)
		self.target_placeholder = tf.placeholder(dtype=target_dtype, shape=target_shape)
		# InputPipeline.__init__(self, sess=sess, queue_data_placeholder=input_placeholder,
		# 					   queue_target_placeholder=target_placeholder, dtypes=[data_dtype, target_dtype],
		# 					   shapes=[data_shape[1:], target_shape[1:]], options=options)
		# dataset = tf.contrib.data.Dataset.from_tensor_slices((input_placeholder, target_placeholder))
		# self.iterator = dataset.make_initializable_iterator()

	def set_dataset_indices(self, dataset_indices: DatasetIndices):
		self.dataset_indices = dataset_indices

	def enqueue_callback(self, sess):
		assert self.dataset_indices is not None  # must be explicitly set
		index_batch_generator = BatchGenerator(sequence=self.dataset_indices.sequence, seq_len=SEQ_LEN,
											   batch_size=BATCH_SIZE,
											   left_context=LEFT_CONTEXT)
		with self.graph.as_default():
			with tf.name_scope("enqueue_pipeline"):
				input_indices = tf.placeholder(shape=(BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN),
											   dtype=tf.string)  # pathes to png files from the central camera
				targets = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
										 dtype=tf.float32)  # seq_len x batch_size x OUTPUT_DIM

				input_images = tf.stack(
					[tf.image.decode_png(tf.read_file(x)) for x in
					 tf.unstack(
						 tf.reshape(input_indices, shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE]))
					 ]
				)
				targets_normalized = (targets - self.dataset_indices.mean) / self.dataset_indices.std

				input_images = tf.cast(input_images, tf.float32) * 2.0 / 255.0 - 1.0
				input_images.set_shape([(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
				video = tf.reshape(input_images, shape=[BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN, HEIGHT, WIDTH, CHANNELS])

				while not self.coord.should_stop():
					# TODO directly tie image processing to dataset
					feed_input_indices, feed_targets = index_batch_generator.next()
					input_batch, target_batch = sess.run([video, targets_normalized],
														 feed_dict={input_indices: feed_input_indices,
																	targets: feed_targets})
					sess.run(self.iterator,
							 feed_dict={self.input_placeholder: input_batch, self.target_batch: target_batch})
					# self.enqueue(sess=sess, curr_data=input_batch, curr_target=target_batch)
					print("Enqueued batch")

	def start_pipeline(self):
		# self.sess.run(self.queue.close(cancel_pending_enqueues=True), options=run_options)
		self.enqueue_thread = threading.Thread(target=self.enqueue_callback, args=[self.sess])
		self.enqueue_thread.isDaemon()
		self.enqueue_thread.start()
