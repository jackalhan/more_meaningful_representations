"""Define the model."""

import tensorflow as tf

from triplet_loss.triplet_loss import batch_all_triplet_loss
from triplet_loss.triplet_loss import batch_hard_triplet_loss, negative_triplet_loss
import numpy as np

def build_fully_connected_model(data, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        data: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    with tf.name_scope('fc') as scope:
        out = tf.contrib.layers.fully_connected(
            data,
            int(data.shape[1]),
            activation_fn=None,
            weights_initializer=tf.glorot_normal_initializer(seed=230), #,tf.truncated_normal_initializer(seed=230, stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
            biases_initializer=tf.constant_initializer(0),
            scope=scope,
            trainable=True,

        )

        data_ = tf.add(out * 1, data, name='add')
        out = tf.nn.l2_normalize(data_, name='out', axis=1)
    return out


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        questions: questions in the batch
        paragraphs: paragraphs in the batch
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    questions = features #tf.slice(features, 0, params.embedding_size)
    paragraphs = labels #tf.slice(features, params.embedding_size, params.embedding_size)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        embeddings = build_fully_connected_model(questions, params)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=embeddings)

    #labels = tf.cast(labels, tf.int64)


    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(paragraphs, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(paragraphs, embeddings, margin=params.margin,
                                       squared=params.squared)
    elif params.triplet_strategy == 'semi_hard':
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(paragraphs, embeddings)
    elif params.triplet_strategy == 'basic_triplet':
        loss = negative_triplet_loss(embeddings, paragraphs, margin=params.margin)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # # -----------------------------------------------------------
    # # METRICS AND SUMMARIES
    # # Metrics for evaluation using tf.metrics (average over whole dataset)
    # # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # # Summaries for training
    # tf.summary.scalar('loss', loss)
    # if params.triplet_strategy == "batch_all":
    #     tf.summary.scalar('fraction_positive_triplets', fraction)

    # Define training step that minimizes the loss with the Adam optimizer
    #optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)
        # check_op = tf.add_check_numerics_ops()
        # _ = tf.Variable(labels, validate_shape=False)
        # tf.Print(_, [_])
        # gvs = optimizer.compute_gradients(loss)
        # capped_gvs = \
        #     [(tf.clip_by_value(grad, -0.1, 0.1), var) if grad != None else (grad, var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def triplet_loss(question_embeddings, paragraph_embeddings, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    anchors, positives, negatives = create_anchor_positive_negative(question_embeddings, paragraph_embeddings)

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)

    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchors - negatives), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    mean_loss = tf.reduce_mean(loss)
    return mean_loss



