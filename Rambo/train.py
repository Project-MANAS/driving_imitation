import os
import argparse
from random import shuffle, randint
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model
from model import Rambo

ap = argparse.ArgumentParser()
ap.add_argument("--train_dir", required=True,
	help="path to the directory where trained models are to be stored")
ap.add_argument("--data_dir", default="./data",
	help="path to the directory where the data is stored")
ap.add_argument("--log_dir", default="./graph",
	help="path to the directory where tensorboard logs are to be written")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
ap.add_argument("--num_channels", type=int, default=4,
	help="number of channels in training images")
ap.add_argument("--rows", type=int, default=192,
	help="number of rows in training images")
ap.add_argument("--cols", type=int, default=256,
	help="number of columns in training images")
ap.add_argument("--num_epochs", type=int, default=10,
	help="number of epochs")
ap.add_argument("--batch_size", type=int, default=32,
	help="size of mini-batch")
ap.add_argument("--shuffle", help="shuffle training data", action="store_true")
args = vars(ap.parse_args())

data_dir = args["data_dir"]


def load_data():
	X_train1 = np.load(os.path.join(data_dir, "X_train_part1.npy"))
	y_train1 = np.load(os.path.join(data_dir, "y_train_part1.npy"))
	X_train2 = np.load(os.path.join(data_dir, "X_train_part2.npy"))
	y_train2 = np.load(os.path.join(data_dir, "y_train_part2.npy"))
	# X_train3 = np.load(os.path.join(data_dir, "X_train_part3.npy"))
	# y_train3 = np.load(os.path.join(data_dir, "y_train_part3.npy"))
	X_train4 = np.load(os.path.join(data_dir, "X_train_part4.npy"))
	y_train4 = np.load(os.path.join(data_dir, "y_train_part4.npy"))
	X_train5 = np.load(os.path.join(data_dir, "X_train_part5.npy"))
	y_train5 = np.load(os.path.join(data_dir, "y_train_part5.npy"))
	X_train6 = np.load(os.path.join(data_dir, "X_train_part6.npy"))
	y_train6 = np.load(os.path.join(data_dir, "y_train_part6.npy"))

	X_train = np.concatenate((X_train1[:3800], X_train2[:13500], X_train4[:1680], X_train5[:3600], X_train6[:6300]), axis=0)
	y_train = np.concatenate((y_train1[:3800], y_train2[:13500], y_train4[:1680], y_train5[:3600], y_train6[:6300]), axis=0)
	X_test = np.concatenate((X_train1[3800:], X_train2[13500:], X_train4[1680:], X_train5[3600:], X_train6[6300:]), axis=0)
	y_test = np.concatenate((y_train1[3800:], y_train2[13500:], y_train4[1680:], y_train5[3600:], y_train6[6300:]), axis=0)

	return X_train[:5019], y_train[:5019], X_test[:200], y_test[:200]
	# return X_train1[:509], y_train1[:509], X_train1[1009:1100], y_train1[1009:1100]


def shuffleDataAndLabelsInPlace(arr1, arr2):
	seed = randint(0, 4294967295)
	prng = RandomState(seed)
	prng.shuffle(arr1)
	prng = RandomState(seed)
	prng.shuffle(arr2)


if __name__ == '__main__':
	gpus = args["gpus"]
	ch = args["num_channels"]
	row = args["rows"]
	col = args["cols"]
	num_epoch = args["num_epochs"]
	batch_size = args["batch_size"] * gpus
	log_dir = args["log_dir"]
	train_dir = args["train_dir"]
	shuffle = args["shuffle"]

	X_train, y_train, X_test, y_test = load_data()

	print "X_train shape:" + str(X_train.shape)
	print "X_test shape:" + str(X_test.shape)
	print "y_train shape:" + str(y_train.shape)
	print "y_test shape:" + str(y_test.shape)

	np.random.seed(1235)
	train_idx_shf = np.random.permutation(X_train.shape[0])

	print "Computing training set mean..."
	X_train_mean = np.mean(X_train, axis=0, keepdims=True)
	
	print "Saving training set mean..."
	np.save(os.path.join(data_dir, 'X_train_mean'), X_train_mean)

	if gpus <= 1:
		print("Training on 1 GPU")
		model = Rambo()
	else:
		print("Training on {} GPUs".format(gpus))
		with tf.device("/cpu:0"):
			# initialize the model
			model = Rambo()

		model = multi_gpu_model(model, gpus=gpus)

	model.compile(optimizer="adam", loss="mse")
	print model.summary()

	model_json = model.to_json()
	with open("./models/rambo.json", "w") as json_file:
		json_file.write(model_json)
	
	# checkpoint
	filepath = os.path.join(train_dir, "rambo-{epoch:02d}-{val_loss:.5f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
	tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
	callbacks_list = [checkpoint, tensorboard]

	iters_train = X_train.shape[0]
	iters_train /= batch_size
	iters_test = X_test.shape[0]
	iters_test /= batch_size

	def my_train_generator():
		while 1:
			for i in range(iters_train):
				if i == iters_train-1:
					idx = train_idx_shf[i*batch_size:]
				else:
					idx = train_idx_shf[i*batch_size:(i+1)*batch_size]
				tmp = X_train[idx].astype('float32')
				tmp -= X_train_mean
				tmp /= 255.0
				yield tmp, y_train[idx]
			
			if shuffle:
				shuffleDataAndLabelsInPlace(X_train, y_train)

	def my_test_generator():
		while 1:
			for i in range(iters_test):
				if i == iters_test-1:
					tmp = X_test[i*batch_size:].astype('float32')
					tmpy = y_test[i*batch_size:]
				else:
					tmp = X_test[i*batch_size:(i+1)*batch_size].astype('float32')
					tmpy = y_test[i*batch_size:(i+1)*batch_size]
				tmp -= X_train_mean
				tmp /= 255.0
				yield tmp, tmpy


	model.fit_generator(my_train_generator(),
		nb_epoch=num_epoch,
		steps_per_epoch=iters_train,
		validation_data=my_test_generator(),
		nb_val_samples=iters_test,
		callbacks=callbacks_list,
		nb_worker=1
	)
