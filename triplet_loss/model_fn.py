"""Define the model."""

import tensorflow as tf

from triplet_loss.triplet_loss import batch_all_triplet_loss
from triplet_loss.triplet_loss import batch_hard_triplet_loss
import numpy as np

def build_model(is_training, data, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        data: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = data
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = 32
    bn_momentum = params.bn_momentum
    channels = [32, 32 * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.shape[1:] == [16, 4, num_channels * 2]

    out = tf.reshape(out, [-1, 16 * 4 * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out
def build_fully_connected_model(is_training, data, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        data: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # l2_data = tf.nn.l2_normalize(
    #     data,
    #     epsilon=1e-12,
    #     name='l2',
    #     axis=0
    # )

    h1 = tf.contrib.layers.fully_connected(
        data,
        params.n_hidden,
        weights_initializer=tf.zeros_initializer(),
        biases_initializer=tf.constant_initializer(0.1),
        trainable=True,
        scope='h1'
    )

    out = tf.contrib.layers.fully_connected(
        h1,
        int(data.shape[1]),
        activation_fn=None,
        weights_initializer=tf.zeros_initializer(),
        biases_initializer=tf.constant_initializer(0.1),
        trainable=True,
        scope='output'
    )

    return out

def build_fully_connected_model(data, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        data: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """

    W_h = tf.get_variable("weights_hidden", shape=[data.shape[1],  params.num_of_units],
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.001), trainable=True)
    b_h = tf.get_variable("bias_hidden", shape=[params.num_of_units],
                          initializer=tf.constant_initializer(0.0), trainable=True)

    W_o = tf.get_variable("weights_out", shape=[params.num_of_units, data.shape[1]],
                          initializer=tf.truncated_normal_initializer(mean=0, stddev=0.001), trainable=True)

    b_o = tf.get_variable("bias_output", shape=[data.shape[1]],
                          initializer=tf.constant_initializer(0.0), trainable=True)


    hidden_layer = tf.add(tf.matmul(data, W_h),b_h)
    hidden_layer = tf.nn.relu(hidden_layer)
    out = tf.add(tf.matmul(hidden_layer, W_o), b_o)

    return out

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        embeddings = build_fully_connected_model(features, params)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=features + embeddings)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

