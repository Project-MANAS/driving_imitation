import math

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import multi_gpu
from dataset.constant import *
from dataset.dataset import get_datasets, DatasetType
from model import Komanda

iter_op, type_ops, mean, var, no_of_iters = get_datasets()

print("Network boi is building")
lr = tf.placeholder_with_default(tf.constant(1e-3, dtype = tf.float32), ())
model = Komanda(BATCH_SIZE, SEQ_LEN, mean, var)
optimizer = tf.train.AdamOptimizer(lr)
train_op = multi_gpu.build(model, optimizer, iter_op, 1)
print("Network boi built")

steering_mse, overall_mse, total_loss = model.output

tf.summary.scalar('steering_mse', steering_mse)
tf.summary.scalar('overall_mse', overall_mse)
tf.summary.scalar('total_loss', total_loss)

summaries = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/train_summary', graph = tf.get_default_graph())
valid_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph = tf.get_default_graph())
test_writer = tf.summary.FileWriter(CHECKPOINT_DIR + '/valid_summary', graph = tf.get_default_graph())

saver = tf.train.Saver(write_version = tf.train.SaverDef.V2)

print("All bois are built")

KEEP_PROB_TRAIN = 0.25


def do_epoch(sess, run_type: DatasetType):
    acc_loss = np.float128(0.0)

    sess.run(type_ops[run_type])
    ops_train = [train_op, total_loss, summaries]
    ops_valid_test = [total_loss, summaries]
    for step in range(no_of_iters[run_type]):
        feed_dict = {}

        if run_type == DatasetType.TRAIN:
            feed_dict.update({model.keep_prob: KEEP_PROB_TRAIN})
            _, loss, summary = session.run(ops_train)
            train_writer.add_summary(summary)
        else:
            loss, summary = session.run(ops_valid_test)

            if run_type == DatasetType.TEST:
                test_writer.add_summary(summary)
            else:
                valid_writer.add_summary(summary)

        acc_loss += loss
        print('Iter', step, '\t', loss)

    avg_loss = acc_loss / no_of_iters[run_type]
    print('Average loss this epoch:', avg_loss)

    return avg_loss


def run_tf_profiling(sess):
    sess.run(type_ops[DatasetType.TRAIN])
    options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for _ in range(10):
        sess.run(train_op, options = options, run_metadata = run_metadata, feed_dict={model.keep_prob: 0.25})

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95, allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
best_validation_score = math.inf

with tf.Session(graph = tf.get_default_graph(), config = config) as session:
    session.run(tf.global_variables_initializer())
    print("All bois are initialized")

    run_tf_profiling(session)
    exit()

    for epoch in range(NUM_EPOCHS):
        print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

        valid_score = do_epoch(session, DatasetType.VALIDATION)
        print("Validation score:", valid_score)

        if valid_score < best_validation_score:
            # saver.save(session, CHECKPOINT_DIR + '/checkpoint')
            best_validation_score = valid_score
            print("SAVED at epoch %d" % epoch)

        if epoch != NUM_EPOCHS - 1:
            print('Komanda boii is training')
            do_epoch(session, DatasetType.TRAIN)

        if epoch == NUM_EPOCHS - 1:
            print("Test:")
            test_score = do_epoch(session, DatasetType.TEST)
            print('Final test loss:', test_score)
