import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.9999


def tower_loss(model, iter_op, wt_dev, op_dev, training):
    inputs = iter_op.get_next()
    model.build(*inputs, op_dev, wt_dev, training)
    losses = tf.get_collection('losses')
    return tf.add_n(losses)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis = 0, values = grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build(model, optimizer, iter_op, num_gpu = 1, training = True, clip_param = None):
    global_step = tf.get_variable('global_step', [], tf.int32, tf.constant_initializer(0), trainable = False)
    if num_gpu == 1:
        with tf.device('/gpu:0'):
            loss = tower_loss(model, iter_op, '/gpu:0', '/gpu:0', training)
            train_op = optimizer.minimize(loss, global_step = global_step)
        return train_op

    tower_grads = []

    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i):
            loss = tower_loss(model, iter_op, '/cpu:0', '/gpu:%d' % i, training)
            gradvars = optimizer.compute_gradients(loss, colocate_gradients_with_ops = True)

            if clip_param is not None:
                gradients, v = zip(*gradvars)
                gradients, _ = tf.clip_by_global_norm(gradients, clip_param)
                gradvars = zip(gradients, v)

            tower_grads.append(gradvars)

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step = global_step)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    return train_op
