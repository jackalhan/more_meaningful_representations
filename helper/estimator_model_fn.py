import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.models import orchestrate_model
from helper.quadratic_loss import euclidean_distance_loss

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: questions in the batch
        labels: paragraphs in the batch
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    questions = features
    paragraphs_ = labels
    # extract labels from paragraphs because we embedded labels and paragraphs into paragraphs tensor [estimator api]
    labels = questions[:, params.embedding_dim:params.embedding_dim + 1]
    questions = questions[:, :params.embedding_dim]

    #labels = tf.cast(labels, dtype=tf.int32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # -----------------------------------------------------------
    embeddings = orchestrate_model(questions, params)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=tf.concat([embeddings, labels], axis=1))

    if params.loss == 'abs_reg_loss':
        paragraphs = tf.nn.l2_normalize(paragraphs_, name='normalize_paragraphs', axis=1)
        loss = euclidean_distance_loss(embeddings, paragraphs, params,
                                        type=params.loss_type)

    else:
        raise ValueError("Loss strategy not recognized: {}".format(params.loss))

    # # -----------------------------------------------------------
    # # METRICS AND SUMMARIES
    # # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # Summaries for training
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)