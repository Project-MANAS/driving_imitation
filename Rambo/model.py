from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Input, concatenate, Dropout

def comma_model(input_layer=None, droput=False, get_model=False):
	if input_layer is None:
		input_layer = Input(batch_shape=[None,192,256,4])
	conv1 = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), padding="same", activation='relu')(input_layer)
	conv2 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
	conv3 = Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
	conv3_f = Flatten()(conv3)
	if droput:
		drop1 = Dropout(rate=0.2)(conv3_f)
		dense1 = Dense(units=100, activation='relu')(drop1)
	else:
		dense1 = Dense(units=512, activation='relu')(conv3_f)

	if get_model:
		if droput:
			drop2 = Dropout(rate=0.2)(dense1)
			output = Dense(units=1)(drop2)
		else:
			output = Dense(units=1)(dense1)
		model = Model(inputs=[input_layer], outputs=[output])
		return model
	else:
		return dense1


def nvidia_model1(input_layer=None, droput=False, get_model=False):
	if input_layer is None:
		input_layer = Input(batch_shape=[None,192,256,4])
	conv1 = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(input_layer)
	conv2 = Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
	conv3 = Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
	conv4 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3)
	conv5 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4)
	conv5_f = Flatten()(conv5)
	if droput:
		drop1 = Dropout(rate=0.2)(conv5_f)
		dense1 = Dense(units=100, activation='relu')(drop1)
		drop2 = Dropout(rate=0.2)(dense1)
		dense2 = Dense(units=50, activation='relu')(drop2)
		drop3 = Dropout(rate=0.2)(dense2)
		dense3 = Dense(units=10, activation='relu')(drop3)
	else:
		dense1 = Dense(units=100, activation='relu')(conv5_f)
		dense2 = Dense(units=50, activation='relu')(dense1)
		dense3 = Dense(units=10, activation='relu')(dense2)

	if get_model:
		if droput:
			drop4 = Dropout(rate=0.2)(dense3)
			output = Dense(units=1)(drop4)
		else:
			output = Dense(units=1)(dense3)
		model = Model(inputs=[input_layer], outputs=[output])
		return model
	else:
		return dense3


def nvidia_model2(input_layer=None, droput=False, get_model=False):
	if input_layer is None:
		input_layer = Input(batch_shape=[None,192,256,4])
	conv1 = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(norm1)
	conv2 = Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
	conv3 = Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
	conv4 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3)
	conv5 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4)
	conv6 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5)
	conv6_f = Flatten()(conv6)
	if droput:
		drop1 = Dropout(rate=0.2)(conv6_f)
		dense1 = Dense(units=100, activation='relu')(drop1)
		drop2 = Dropout(rate=0.2)(dense1)
		dense2 = Dense(units=50, activation='relu')(drop2)
		drop3 = Dropout(rate=0.2)(dense2)
		dense3 = Dense(units=10, activation='relu')(drop3)
	else:
		dense1 = Dense(units=100, activation='relu')(conv6_f)
		dense2 = Dense(units=50, activation='relu')(dense1)
		dense3 = Dense(units=10, activation='relu')(dense2)

	if get_model:
		if droput:
			drop4 = Dropout(rate=0.2)(dense3)
			output = Dense(units=1)(drop4)
		else:
			output = Dense(units=1)(dense3)
		model = Model(inputs=[input_layer], outputs=[output])
		return model
	else:
		return dense3


def Rambo():
	input_layer = Input(batch_shape=[None,192,256,4])
	comma_out = comma_model(input_layer)
	nvidia1_out = nvidia_model1(input_layer)
	nvidia2_out = nvidia_model2(input_layer)
	merged_out = concatenate([comma_out,nvidia1_out,nvidia2_out])
	output = Dense(units=1)(merged_out)
	model = Model(inputs=[input_layer], outputs=[output])
	return model