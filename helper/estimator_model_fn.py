import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.models import orchestrate_model
from helper.loss import euclidean_distance_loss
from helper.utils import load_embeddings, calculate_recalls


def recall_fn(base_data_path, params, question_embeddings):
    # Evaluate the model's recall on the test set
    # 5k questions, ~20k paragraphs, can not be handled in estimator api (number of questions and number of paragraphs are not same)
    # In order to accelarete the calculation, I prefer doing the following
    all_paragraphs = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                              params.files["test_subset_recall"][
                                                                  "paragraph_embeddings"])))

    normalized_all_paragraphs = tf.nn.l2_normalize(all_paragraphs,
                                                   name='normalized_all_paragraph_embeddings',
                                                   axis=1)
    all_paragraphs_norm = tf.reduce_mean(tf.norm(normalized_all_paragraphs, axis=1))
    subset_labels = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                             params.files["test_subset_recall"][
                                                                 "question_labels"])))
    subset_labels = tf.reshape(subset_labels, [-1, 1])
    recalls, normalized_recalls = calculate_recalls(question_embeddings, normalized_all_paragraphs, subset_labels, params)
    avg_recall = tf.reduce_mean(normalized_recalls)
    return avg_recall, normalized_recalls

def model_fn(features, labels, mode, params, config):
    """Model function for tf.estimator

    Args:
        features: questions in the batch
        labels: paragraphs in the batch
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """

    base_data_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])
    questions = features
    global_step = tf.train.get_global_step()
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # -----------------------------------------------------------
    embeddings = orchestrate_model(questions, params)
    #question_embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    paragraph = tf.nn.l2_normalize(labels, name='normalized_paragraph_embeddings', axis=1)
    #paragraph_embedding_mean_norm = tf.reduce_mean(tf.norm(paragraph, axis=1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=embeddings)

    if params.loss['name'] == 'abs_reg_loss':
        _loss = euclidean_distance_loss(embeddings, paragraph, params)
    else:
        raise ValueError("Loss strategy not recognized: {}".format(params.loss['name']))

    tf.losses.add_loss(_loss)
    loss = tf.losses.get_total_loss()
    # # -----------------------------------------------------------
    # # METRICS AND SUMMARIES
    # # Metrics for evaluation using tf.metrics (average over whole dataset)
    # with tf.variable_scope("metrics"):

    tf.summary.scalar('loss', loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        global_step_ = tf.Print(global_step, [global_step], message="Value of global step")
        avg_recall, normalized_recalls = recall_fn(base_data_path, params, embeddings)
        loss_over_recall_top_1 = tf.cast(loss, dtype=tf.float64) + tf.keras.backend.epsilon() / normalized_recalls[0]
        tf.summary.scalar("avg_recall", avg_recall)
        tf.summary.scalar("recall_top_1", normalized_recalls[0])
        tf.summary.scalar("recall_top_2", normalized_recalls[1])
        tf.summary.scalar("loss/recall_top_1", loss_over_recall_top_1)
        eval_metric_ops= {"recall_top_1": tf.metrics.mean(normalized_recalls[0])}
        eval_metric_ops["recall_top_2"] = tf.metrics.mean(normalized_recalls[1])
        eval_metric_ops["avg_recall"] = tf.metrics.mean(avg_recall)
        eval_metric_ops["loss/recall_top_1"] = tf.metrics.mean(loss_over_recall_top_1)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    if params.optimizer['name'] == 'Adam':
        optimizer = tf.train.AdamOptimizer(params.optimizer['learning_rate'])
    else:
        raise ValueError("Optimizer is not recognized: {}".format(params.optimizer['name']))


    #tf.summary.scalar("global_step", global_step)

    if params.optimizer['batch_norm']:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)