import math

import multi_gpu
import numpy as np
import tensorflow as tf
import tqdm
from dataset.constant import *
from dataset.dataset import get_datasets, DatasetType
from model import Komanda
from tensorflow.python.client import timeline

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


def do_train_epoch(sess):
    sess.run(type_ops[DatasetType.TRAIN])
    accumulated_loss = np.float128(0.0)
    ops = [train_op, total_loss, summaries]
    feed_dict = {model.keep_prob: KEEP_PROB_TRAIN}

    for _ in tqdm.trange(no_of_iters[DatasetType.TRAIN]):
        _, loss, summary = sess.run(ops, feed_dict)
        train_writer.add_summary(summary)
        accumulated_loss += loss

    avg_loss = accumulated_loss / no_of_iters[DatasetType.TRAIN]
    print('Average training loss:', avg_loss)
    return avg_loss


def do_test_epoch(sess, run_type: DatasetType):
    accumulated_loss = np.float128(0.0)
    sess.run(type_ops[run_type])
    ops = [total_loss, summaries]
    summary_writer = test_writer if DatasetType.TEST == run_type else valid_writer

    for _ in tqdm.trange(no_of_iters[run_type]):
        loss, summary = sess.run(ops)
        summary_writer.add_summary(summary)
        accumulated_loss += loss

    avg_loss = accumulated_loss / no_of_iters[run_type]
    print('Average loss:', avg_loss)
    return avg_loss


def run_tf_profiling(sess):
    sess.run(type_ops[DatasetType.TRAIN])
    options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for _ in range(10):
        sess.run(train_op, options = options, run_metadata = run_metadata, feed_dict = {model.keep_prob: 0.25})

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

    for epoch in range(NUM_EPOCHS):
        print("Starting epoch %d / %d" % (epoch, NUM_EPOCHS))

        print("Validation:")
        valid_score = do_test_epoch(session, DatasetType.VALIDATION)

        if valid_score < best_validation_score:
            # saver.save(session, CHECKPOINT_DIR + '/checkpoint')
            best_validation_score = valid_score
            print("SAVED at epoch %d" % epoch)

        if epoch != NUM_EPOCHS - 1:
            print('Komanda boii is training')
            do_train_epoch(session)

        if epoch == NUM_EPOCHS - 1:
            print("Test:")
            test_score = do_test_epoch(session, DatasetType.TEST)
            print('Final test loss:', test_score)
