import threading

import tensorflow as tf

from manas.ai.dataset.input_pipeline import InputPipeline, InputPipelineOptions
from manas.ai.planning.komanda.dataset.dataset import *


class PipelineOptions(InputPipelineOptions):
	def __init__(self):
		super().__init__()
		self.min_after_dequeue = 1
		self.num_threads = 2
		self.small_safety_margin = 2
		self.batch_size = 1
		self.enqueue_size = 0


class Pipeline(InputPipeline):
	def __init__(self, graph, sess, batch_container: BatchContainer, options: PipelineOptions):
		self.graph = graph
		self.sess = sess
		self.dataset_indices = None

		output_shape = batch_container.dataset.output_shapes
		output_types = batch_container.dataset.output_types

		# TODO Use single placeholder to hold all types
		input_placeholder = tf.placeholder(dtype=output_types[0], shape=output_shape[0])
		target_placeholder = tf.placeholder(dtype=output_types[1], shape=output_shape[1])

		# 	TODO Pass iterator instead
		self.batch_count = batch_container.count
		self._next_op = batch_container.dataset.make_one_shot_iterator().get_next()
		batch = self.sess.run([self._next_op])
		print("Batch shape: ", len(batch))

		super().__init__(sess, input_placeholder, target_placeholder, output_types, output_shape, options)

	# InputPipeline.__init__(self, sess=sess, queue_data_placeholder=input_placeholder,
	# 					   queue_target_placeholder=target_placeholder, dtypes=[data_dtype, target_dtype],
	# 					   shapes=[data_shape[1:], target_shape[1:]], options=options)

	def set_dataset_indices(self, dataset_indices: DatasetIndices):
		self.dataset_indices = dataset_indices

	def enqueue_callback(self, sess):
		while not self.coord.should_stop():
			# TODO directly tie image processing to dataset
			batch = self.sess.run(self._next_op)
			self.enqueue(sess=sess, curr_data=batch[0], curr_target=batch[1])
			print("Enqueued batch. Size now %d" % sess.run(self.queue.size()))

	def start_pipeline(self):
		# self.sess.run(self.queue.close(cancel_pending_enqueues=True), options=run_options)
		self.enqueue_thread = threading.Thread(target=self.enqueue_callback, args=[self.sess])
		self.enqueue_thread.isDaemon()
		self.enqueue_thread.start()
