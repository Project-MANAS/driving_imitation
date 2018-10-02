import tensorflow as tf
import tfutils as tu

from dataset.constant import *


class Komanda:
    def __init__(self, batch_size, seq_len, dataset_mean = 0, dataset_variance = 1):
        self.keep_prob = tf.placeholder_with_default(1.0, (), "keep_prob")
        self.aux_cost_weight = tf.placeholder_with_default(0.1, (), "aux_cost_weight")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mean = dataset_mean
        self.var = dataset_variance
        self.info = {}
        self.output = {}
        self.TOWER_NAME = "KomandaBoi"

    def flatten(self, inputs):
        # inputs is of size (batch_size, channels, seq_len, height, width) --> (batch_size * seq_len, c*h*w)
        channels_last = tf.transpose(inputs, [0, 2, 3, 4, 1])
        flattened = tf.reshape(channels_last, [self.batch_size * self.seq_len, -1])
        return flattened

    def vision_block(self, video, op_dev, wt_dev):
        # video is of shape (batch_size, channels, seq_len, height, width)
        activation_fn = tf.nn.relu

        net1 = tu.layers.conv3d(video, 'conv3d_1', 64, [3, 12, 12], [1, 6, 6], "VALID", [1, 1, 1], "NCDHW", wt_dev,
                                op_dev)
        net1 = activation_fn(net1)
        net1 = tf.nn.dropout(net1, self.keep_prob)

        net2 = tu.layers.conv3d(net1, 'conv3d_2', 64, [2, 5, 5], [1, 2, 2], "VALID", [1, 1, 1], "NCDHW", wt_dev, op_dev)
        net2 = activation_fn(net2)
        net2 = tf.nn.dropout(net2, self.keep_prob)

        net3 = tu.layers.conv3d(net2, 'conv3d_3', 64, [2, 5, 5], [1, 1, 1], "VALID", [1, 1, 1], "NCDHW", wt_dev, op_dev)
        net3 = activation_fn(net3)
        net3 = tf.nn.dropout(net3, self.keep_prob)

        net4 = tu.layers.conv3d(net3, 'conv3d_4', 64, [2, 5, 5], [1, 1, 1], "VALID", [1, 1, 1], "NCDHW", wt_dev, op_dev)
        net4 = activation_fn(net4)
        net4 = tf.nn.dropout(net4, self.keep_prob)

        flat = self.flatten(net4)
        net5 = tu.layers.dense(flat, 'fc_1', 1024, True, wt_dev, op_dev)
        net5 = activation_fn(net5)
        net5 = tf.nn.dropout(net5, self.keep_prob)

        net6 = tu.layers.dense(net5, 'fc_2', 512, True, wt_dev, op_dev)
        net6 = activation_fn(net6)
        net6 = tf.nn.dropout(net6, self.keep_prob)

        net7 = tu.layers.dense(net6, 'fc_3', 256, True, wt_dev, op_dev)
        net7 = activation_fn(net7)
        net7 = tf.nn.dropout(net7, self.keep_prob)

        net8 = tu.layers.dense(net7, 'fc_4', 128, True, wt_dev, op_dev)
        net8 = activation_fn(net8)
        net8 = tf.nn.dropout(net8, self.keep_prob)

        aux1 = self.flatten(net1[:, :, -self.seq_len:, :, :])
        aux1 = tu.layers.dense(aux1, 'aux_1', 128, True, wt_dev, op_dev)
        aux2 = self.flatten(net2[:, :, -self.seq_len:, :, :])
        aux2 = tu.layers.dense(aux2, 'aux_2', 128, True, wt_dev, op_dev)
        aux3 = self.flatten(net3[:, :, -self.seq_len:, :, :])
        aux3 = tu.layers.dense(aux3, 'aux_3', 128, True, wt_dev, op_dev)
        aux4 = self.flatten(net4[:, :, -self.seq_len:, :, :])
        aux4 = tu.layers.dense(aux4, 'aux_4', 128, True, wt_dev, op_dev)

        net9 = activation_fn(aux1 + aux2 + aux3 + aux4 + net8)
        unflat = tf.reshape(net9, [self.batch_size, self.seq_len, -1])
        return tf.contrib.layers.layer_norm(unflat)

    def build(self, video, targets, op_dev, wt_dev, training = True):
        conv_net = self.vision_block(video, op_dev, wt_dev)
        conv_net = tf.nn.dropout(conv_net, self.keep_prob)
        conv_net = tf.transpose(conv_net, [1, 0, 2])

        lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(RNN_LAYERS, RNN_SIZE)
        outputs, _ = lstm_cell(conv_net, training = training)  # (seq_len, batch, RNN_SIZE)
        outputs = tf.reshape(outputs, [self.seq_len * self.batch_size, -1])
        outputs = tu.layers.dense(outputs, 'prediction', OUTPUT_DIM, True, wt_dev, op_dev)
        outputs = tf.reshape(outputs, [self.seq_len, self.batch_size, -1])

        overall_mse = tf.reduce_mean(tf.squared_difference(targets, outputs), name = 'overall_mse')
        steering_mse = tf.reduce_mean(tf.squared_difference(targets[:, :, 0], outputs[:, :, 0]), name = 'steering_mse')
        total_loss = tf.identity(self.aux_cost_weight * overall_mse + steering_mse, name = 'total_loss')
        tf.add_to_collection('losses', total_loss)

        steering_prediction = outputs[:, :, 0] * self.var + self.mean
        self.info = {'steering_prediction': steering_prediction}
        self.output = (steering_mse, overall_mse, total_loss)
