"""Define the model."""

import tensorflow as tf

from triplet_loss.triplet_loss import batch_all_triplet_loss
from triplet_loss.triplet_loss import batch_hard_triplet_loss, negative_triplet_loss
from triplet_loss.quadratic_loss import euclidean_distance_loss

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
            int(params.embedding_dim),
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=params.seed,stddev=0.1), #,tf.truncated_normal_initializer(seed=230, stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(params.l2_regularizer),
            biases_initializer=tf.constant_initializer(0),
            scope=scope,
            trainable=True,

        )

        data_ = tf.add(out * params.scaling_factor, data, name='add')
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
    questions = features
    paragraphs = labels
    # extract labels from paragraphs because we embedded labels and paragraphs into paragraphs tensor [estimator api]
    labels = questions[:, params.embedding_dim:params.embedding_dim + 1]
    questions = questions[:, :params.embedding_dim]
    labels = tf.cast(labels, dtype=tf.int32)

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


    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(paragraphs, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(paragraphs, embeddings, labels, margin=params.margin,
                                       squared=params.squared)
    elif params.triplet_strategy == 'semi_hard':
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.squeeze(labels), embeddings)
    elif params.triplet_strategy == 'basic_triplet':
        loss = negative_triplet_loss(embeddings, paragraphs, margin=params.margin)
    elif params.triplet_strategy == 'quadratic_reg_loss':
        loss = euclidean_distance_loss(embeddings, paragraphs)
    elif params.triplet_strategy == 'abs_reg_loss':
        loss = euclidean_distance_loss(embeddings, paragraphs, type='abs')
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
        #eval_metric_ops = {"recall_at_k": tf.metrics.mean(calculate_recalls(questions, paragraphs, labels, params))}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

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