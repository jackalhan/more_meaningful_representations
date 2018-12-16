import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.models import orchestrate_model
from helper.loss import euclidean_distance_loss, euclidean_distance
from helper.utils import get_file_key_names_for_execution, load_embeddings, evaluation_metrics, pairwise_expanded_cosine_similarities, calculate_top_k, pairwise_euclidean_distances

def eval_metrics_fn(base_data_path, params, before_model_embeddings, after_model_embeddings, KN_FILE_NAMES):
    """
    # Evaluate the model's recall on the test set
    # 5k questions, ~20k paragraphs, can not be handled in estimator api (number of questions and number of paragraphs are not same)
    # In order to accelarete the calculation, I prefer doing the following
    """

    all_targets = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                          params.files[params.executor["recall_calculation_for"] + "_subset_recall"]["all_" + KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]])))


    normalized_all_targets = tf.nn.l2_normalize(all_targets,
                                                   name='normalized_all_targets_embeddings',
                                                   axis=1)


    subset_labels = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                             params.files[params.executor[
                                                                              "recall_calculation_for"] + "_subset_recall"][
                                                                  KN_FILE_NAMES["KN_SOURCE_LABELS"]])))
    subset_labels = tf.reshape(subset_labels, [-1, 1])

    # AVG RECALLS FOR ALL RECALL_TOPS
    eval_metrics_after = evaluation_metrics(after_model_embeddings,
                                                        normalized_all_targets,
                                                        subset_labels,
                                                        params,
                                                        distance_type=params.executor["distance_type"])
    eval_metrics_before = None
    if params.executor["is_debug_mode"]:
        eval_metrics_before = evaluation_metrics(before_model_embeddings,
                                                             normalized_all_targets,
                                                             subset_labels,
                                                             params,
                                                             distance_type=params.executor[
                                                                                            "distance_type"])

    return eval_metrics_after, eval_metrics_before

def extract_metrics_ops(eval_metrics_after, eval_metrics_before):
    eval_metric_ops = {}
    for fn_name, ks in eval_metrics_after.items():
        for k, v in ks.items():
            if eval_metrics_before is not None:
                eval_metric_ops[fn_name + '_' + str(k)] = tf.reshape(eval_metrics_before[fn_name][k], (1,))
                #eval_metric_ops[fn_name + '_' + str(k)] = tf.reshape(v, (1,))
            else:
                eval_metric_ops[fn_name + '_' + str(k)] = tf.metrics.mean(v)
                tf.summary.scalar(fn_name + '_' + str(k), v)
    return eval_metric_ops
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
    KN_FILE_NAMES = get_file_key_names_for_execution(params)
    base_data_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])
    base_data_path = os.path.join(base_data_path, KN_FILE_NAMES['DIR'])


    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    source_embeddings = features['source_embeddings']
    if params.model['model_type'].lower() == 'conv':
        source_baseline_embeddings = features['baseline_source_embeddings']
    else:
        source_baseline_embeddings = features['source_embeddings']
    # -----------------------------------------------------------
    after_model_embeddings = orchestrate_model(source_embeddings, source_baseline_embeddings, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        before_model_embeddings = tf.nn.l2_normalize(source_baseline_embeddings,
                                                     name='normalized_before_model_source_embeddings', axis=1)
        if params.executor["is_debug_mode"]:
            normalized_output = tf.nn.l2_normalize(after_model_embeddings, axis=1)
            tf.contrib.layers.summarize_activation(normalized_output)
            eval_metrics_after,  eval_metrics_before = eval_metrics_fn(base_data_path,
                                                                      params,
                                                                      before_model_embeddings,
                                                                      after_model_embeddings,
                                                                      KN_FILE_NAMES)

            eval_metric_ops = extract_metrics_ops(eval_metrics_after, eval_metrics_before)
        else:
            eval_metric_ops = after_model_embeddings

        return tf.estimator.EstimatorSpec(mode=mode, predictions=eval_metric_ops)

    if params.loss['require_l2_norm']:
        after_model_embeddings = tf.nn.l2_normalize(after_model_embeddings, axis=1)
    tf.contrib.layers.summarize_activation(after_model_embeddings)

    #if params.model['model_type'].lower() == 'conv':
    targets = labels['target_embeddings']
    labels = tf.cast(labels['target_labels'], tf.float32)

    if params.loss['require_l2_norm']:
        targets = tf.nn.l2_normalize(targets, name='normalized_target_embeddings', axis=1)
    else:
        labels = tf.cast(labels, tf.int32)
    # paragraph_embedding_mean_norm = tf.reduce_mean(tf.norm(paragraph, axis=1))
    if params.loss['name'] == 'abs_reg_loss':
        _loss = euclidean_distance_loss(after_model_embeddings, targets, params, labels)
    else:
        raise ValueError("Loss strategy not recognized: {}".format(params.loss['name']))
    tf.losses.add_loss(_loss)
    loss = tf.losses.get_total_loss()
    # # -----------------------------------------------------------
    # # METRICS AND SUMMARIES
    # # Metrics for evaluation using tf.metrics (average over whole dataset)
    # with tf.variable_scope("metrics"):
    # logging_hook = tf.train.LoggingTensorHook({"step": global_step}, every_n_iter=1)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:

        before_model_embeddings = tf.nn.l2_normalize(source_baseline_embeddings,
                                                     name='normalized_before_model_source_embeddings', axis=1)

        eval_metrics_after, eval_metrics_before = eval_metrics_fn(base_data_path,
                                                                                        params,
                                                                                        before_model_embeddings,
                                                                                        after_model_embeddings,
                                                                                        KN_FILE_NAMES)
        eval_metric_ops = extract_metrics_ops(eval_metrics_after, eval_metrics_before)
        return tf.estimator.EstimatorSpec(mode, loss= loss, eval_metric_ops=eval_metric_ops)

    global_step = tf.train.get_global_step()

    if params.optimizer['name'].lower() == 'adam':
        optimizer = tf.train.AdamOptimizer(params.optimizer['learning_rate']) #tf.contrib.optimizer_v2.AdamOptimizer(params.optimizer['learning_rate']) #
    elif params.optimizer['name'].lower() == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(params.optimizer['learning_rate'])
        tf.summary.scalar("current_learning_rate", optimizer._learning_rate)
    elif params.optimizer['name'].lower() == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(params.optimizer['learning_rate'])
        tf.summary.scalar("current_learning_rate", optimizer._learning_rate)
    elif params.optimizer['name'].lower() == 'exponential':
        learning_rate = tf.train.exponential_decay(params.optimizer['learning_rate'], global_step,
                                                   100000, 0.96, staircase=True)
        tf.summary.scalar("current_learning_rate", learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("Optimizer is not recognized: {}".format(params.optimizer['name']))
    """
    SPECIAL FOR ADAM: START
    """
    # tensorflow calculates the
    # And the real decaying lr_t is computed as an intermediate result inside the computing function. You seems to have to compute it by yourself.
    # lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    #tf.summary.scalar("learning_rate", optimizer._get_per_graph_state().get_hyper("learning_rate"))
    #beta1_power, beta2_power = optimizer._get_beta_accumulators()
    # beta1 = tf.get_variable("beta1_power")
    # beta2 = tf.get_variable("beta2_power")
    # lr = tf.get_variable("learning_rate")
    # # tf.summary.scalar("beta1_power", beta1_power)
    # # tf.summary.scalar("beta2_power", beta2_power)
    # current_lr = (lr * tf.sqrt(1 - beta1) / (1 - beta2))
    # tf.summary.scalar("current learning rate", current_lr)
    """
    SPECIAL FOR ADAM: END
    """
    #tf.summary.scalar("global_step", global_step)

    if params.optimizer['batch_norm']:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)