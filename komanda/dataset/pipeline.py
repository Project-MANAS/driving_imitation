import matplotlib.pyplot as plt
import tensorflow as tf
from imgaug import augmenters as iaa

from manas.ai.dataset.input_pipeline import InputPipeline, InputPipelineOptions
from manas.ai.planning.komanda.dataset.dataset import *


class PipelineOptions(InputPipelineOptions):
	def __init__(self):
		super().__init__()
		self.min_after_dequeue = 1
		self.num_threads = 11
		self.small_safety_margin = 0
		self.batch_size = 1
		self.enqueue_size = 0


class Pipeline(InputPipeline):
	def __init__(self, graph, sess, batch_container: BatchContainer, options: PipelineOptions):
		output_shape = batch_container.dataset.output_shapes
		output_types = batch_container.dataset.output_types
		# TODO Use single placeholder to hold all types
		input_placeholder = tf.placeholder(dtype=output_types[0], shape=output_shape[0])
		target_placeholder = tf.placeholder(dtype=output_types[1], shape=output_shape[1])
		super().__init__(sess, input_placeholder, target_placeholder, output_types, output_shape, options)

		# self.enqueue_thread = threading.Thread(target=self.enqueue_callback, args=[self.sess])
		self.graph = graph

		# TODO Pass iterator instead
		self.batch_count = batch_container.count
		self._next_op = batch_container.dataset.make_one_shot_iterator().get_next()

		sometimes = lambda aug: iaa.Sometimes(p=0.5, then_list=aug, deterministic=True, random_state=0)
		self.seq = iaa.Sequential(
			[
				sometimes(
					iaa.Crop(percent=(0.0, 0.05))
				),
				sometimes(
					iaa.Affine(scale={"x": (1.05, 1.1), "y": (1.05, 1.1)},
							   translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
							   rotate=(-4, 4),
							   shear=(-4, 4),
							   order=[0, 1],
							   cval=(0.01, 1.0), deterministic=True, random_state=0)
				),
				iaa.SomeOf((1, 4),
						   [
							   sometimes(
								   iaa.Superpixels(p_replace=(0.0001, 0.02), n_segments=(20, 40),
												   deterministic=True, random_state=0)
							   ),
							   iaa.OneOf([
								   iaa.GaussianBlur((0, 3.0), deterministic=True, random_state=0),
								   iaa.MedianBlur(k=(3, 5), deterministic=True, random_state=0),
							   ]),
							   iaa.Sharpen(alpha=(0, 0.25), lightness=(0.25, 1.25), deterministic=True, random_state=0),
							   iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0), deterministic=True, random_state=0),
							   sometimes(
								   iaa.OneOf([
									   iaa.EdgeDetect(alpha=(0, 0.05), deterministic=True, random_state=0),
									   iaa.DirectedEdgeDetect(alpha=(0, 0.05), direction=(0.0, 1.0), deterministic=True,
															  random_state=0),
								   ])
							   ),
							   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5,
														 deterministic=True, random_state=0),
							   iaa.Dropout((0.0001, 0.001), per_channel=0.5, deterministic=True, random_state=0),
							   iaa.Add((-20, 20), per_channel=0.5, deterministic=True, random_state=0),
							   iaa.Multiply((0.95, 1.05), per_channel=0.5, deterministic=True, random_state=0),
							   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5, deterministic=True,
														 random_state=0),
							   sometimes(
								   iaa.ElasticTransformation(alpha=(0.5, 2.0), sigma=1.25, deterministic=True,
															 random_state=0)
							   )
						   ],
						   deterministic=True, random_state=0
						   )
			],
			random_order=True,
			deterministic=True
		)

	def enqueue_callback(self, sess, thread_index):
		print('Enqueue thread %d: Initiated' % thread_index)
		while not self.coord.should_stop():  # TODO Loop only as long as session is open
			with tf.name_scope("pipeline_enqueue_%d" % thread_index):
				print('Enqueue thread %d: Batch fetched' % thread_index)
				batch = sess.run(self._next_op)
				# One thread does not augment to prevent any CPU augmentation bottlenecks
				if thread_index == 1:
					self.enqueue(sess=sess, curr_data=batch[0], curr_target=batch[1])
				else:
					batch_aug_gen = [self.generated_augmentations(sequence) for sequence in batch[0]]
					for batch_index in range(N_AUG):
						augmented_batch = [next(sequence_aug_gen) for sequence_aug_gen in batch_aug_gen]
						print('Enqueue thread %d: Produced augmented batch %d (Size now %d)'
							  % (thread_index, batch_index, sess.run(self.queue_size_op)))
						self.enqueue(sess=sess, curr_data=augmented_batch, curr_target=batch[1])

	def start_pipeline(self):
		self.spawn_enqueue_threads(self.enqueue_callback)

	def generated_augmentations(self, sequence):
		assert len(sequence) % N_CAMS == 0
		sequence_batch = np.split(sequence, len(sequence))

		while True:
			augmented_enumeration = self.seq.augment_batches(sequence_batch)
			# sequence_enumerations[i] = [self.seq.augment_image(image) for image in sequence_enumerations[i]]
			# augmented_enumerations.append([self.seq.augment_image(image) for image in sequence])
			self.seq.reseed(deterministic_too=True)
			augmented_enumeration = np.stack(augmented_enumeration, axis=1)[0]

			# self.display_augmentation(augmented_enumeration[0])

			# Normalize augmented images
			yield augmented_enumeration.astype(np.float32) * 2.0 / 255.0 - 1.0

	@staticmethod
	def display_augmentation(augmented_enumerations):
		plt.figure(1)
		plt.ion()
		for index, image in enumerate(augmented_enumerations):
			subplt = plt.subplot(int(len(augmented_enumerations) / N_CAMS), N_CAMS, index + 1)
			subplt.set_xticklabels([])
			subplt.set_yticklabels([])
			plt.imshow(image)
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
		plt.show()
		plt.pause(.1)
