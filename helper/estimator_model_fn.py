import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.models import orchestrate_model
from helper.loss import euclidean_distance_loss, euclidean_distance
from helper.utils import get_file_key_names_for_execution, load_embeddings, calculate_recalls, pairwise_expanded_cosine_similarities, calculate_recall_top_k, pairwise_euclidean_distances

def recall_fn(base_data_path, params, before_model_embeddings, after_model_embeddings, KN_FILE_NAMES):
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


    all_targets_norm = tf.reduce_mean(tf.norm(normalized_all_targets, axis=1))



    subset_labels = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                             params.files[params.executor[
                                                                              "recall_calculation_for"] + "_subset_recall"][
                                                                  KN_FILE_NAMES["KN_SOURCE_LABELS"]])))
    subset_labels = tf.reshape(subset_labels, [-1, 1])




    # AVG RECALLS FOR ALL RECALL_TOPS
    recalls_after_model, normalized_recalls_after_model = calculate_recalls(after_model_embeddings,
                                                                            normalized_all_targets,
                                                    subset_labels,
                                                    params,
                                                    distance_type=params.executor["distance_type"])

    avg_recall_after_model = tf.reduce_mean(normalized_recalls_after_model)

    are_founds_before, closest_labels_before, distances_before, \
    are_founds_after, closest_labels_after, distances_after,\
    avg_recall_before_model,normalized_recalls_before_model,\
    distance_from_before_model_q_to_p,distance_from_after_model_q_to_p, delta_before_after_model = tf.constant(0, dtype=tf.float32), \
                                                                                                   tf.constant(0, dtype=tf.float32), \
                                                                                                   tf.constant(0,dtype=tf.float32), \
                                                                                                   tf.constant(0,dtype=tf.float32), \
                                                                                                   tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),\
                                                                                                      tf.constant(0, dtype=tf.float32),
    if params.executor["is_debug_mode"]:


        target_embeddings = tf.constant(load_embeddings(os.path.join(base_data_path,
                                                                        params.files[params.executor[
                                                                                         "recall_calculation_for"] + "_subset_recall"][
                                                                            KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]])))

        normalized_target_embeddings = tf.nn.l2_normalize(target_embeddings,
                                                             name='normalized_target_embeddings',
                                                             axis=1)



        # ground truth delta before model
        # if params.model['model_type'].lower() == 'conv':
        #     distance_from_before_model_q_to_p = tf.zeros([1,1], tf.float32)
        # else:
        distance_from_before_model_q_to_p = euclidean_distance(before_model_embeddings, normalized_target_embeddings)

        GL_distance_from_before_model_q_to_p = distance_from_before_model_q_to_p


        # ground truth delta after model
        distance_from_after_model_q_to_p = euclidean_distance(after_model_embeddings, normalized_target_embeddings)

        # delta before model - after model to ground truth
        delta_before_after_model = distance_from_before_model_q_to_p - distance_from_after_model_q_to_p

        # if params.model['model_type'].lower() == 'conv':
        #     recalls_before_model, normalized_recalls_before_model = tf.zeros([1, 1], tf.float32), tf.zeros([1, 1], tf.float32)
        # else:
        recalls_before_model, normalized_recalls_before_model = calculate_recalls(before_model_embeddings,
                                                                                        normalized_all_targets,
                                                                                        subset_labels,
                                                                                        params,
                                                                                        distance_type=params.executor[
                                                                                            "distance_type"])


        avg_recall_before_model = tf.reduce_mean(normalized_recalls_before_model)

        if params.executor["distance_type"] == 'cosine':
            # if params.model['model_type'].lower() == 'conv':
            #     _, _labels_before, ___, _scores_before = tf.zeros([1, 1], tf.float32),tf.zeros([1, 1], tf.float32),tf.zeros([1, 1], tf.float32),tf.zeros([1, 1], tf.float32)
            # else:
            _, _labels_after, ___, _scores_after = pairwise_expanded_cosine_similarities(
                                                                                            after_model_embeddings,
                                                                                            subset_labels,
                                                                                            normalized_all_targets)

        else:
            # if params.model['model_type'].lower() == 'conv':
            #     _scores_before = tf.zeros([1, 1], tf.float32)
            # else:
            _scores_before = pairwise_euclidean_distances(before_model_embeddings, normalized_all_targets)
            _labels_before = subset_labels

            _scores_after = pairwise_euclidean_distances(after_model_embeddings, normalized_all_targets)
            _labels_after = subset_labels

        avg_recall_after_model = tf.tile([avg_recall_after_model], [tf.shape(after_model_embeddings)[0]])
        normalized_recalls_after_model = tf.reshape(normalized_recalls_after_model, (1,-1))
        normalized_recalls_after_model = tf.tile(normalized_recalls_after_model, (tf.shape(after_model_embeddings)[0],1))

        # if params.model['model_type'].lower() == 'conv':
        #     avg_recall_before_model = tf.zeros([1, 1], tf.float32)
        #     normalized_recalls_before_model = tf.zeros([1, 1], tf.float32)
        # else:
        avg_recall_before_model = tf.tile([avg_recall_before_model],
                                          [tf.shape(before_model_embeddings)[0]])
        normalized_recalls_before_model = tf.reshape(normalized_recalls_before_model, (1, -1))
        normalized_recalls_before_model = tf.tile(normalized_recalls_before_model, (tf.shape(before_model_embeddings)[0], 1))
        for _k in range(1, params.executor["debug_top_k"]+1):
            # if params.model['model_type'].lower() == 'conv':
            #     _are_founds_before, _closest_labels_before, _distances_before = tf.zeros([1, 1], tf.float32), tf.zeros([1, 1], tf.float32), tf.zeros([1, 1], tf.float32)
            # else:
            _are_founds_before, _closest_labels_before, _distances_before = calculate_recall_top_k(_scores_before,
                                                                                                _labels_before, _k,
                                                                                                params.executor[
                                                                                                    "distance_type"])
            _are_founds_before = tf.reshape(_are_founds_before, [-1, 1])
            _closest_labels_before = tf.reshape(_closest_labels_before, [-1, _k])
            _distances_before = tf.reshape(_distances_before, [-1, _k])


            _are_founds_after, _closest_labels_after, _distances_after = calculate_recall_top_k(_scores_after, _labels_after, _k, params.executor["distance_type"])
            _are_founds_after = tf.reshape(_are_founds_after, [-1, 1])
            _closest_labels_after = tf.reshape(_closest_labels_after, [-1, _k])
            _distances_after = tf.reshape(_distances_after, [-1, _k])

            if _k < 2:
                # if params.model['model_type'].lower() == 'conv':
                #     are_founds_before = tf.zeros([1, 1], tf.float32)
                #     closest_labels_before = tf.zeros([1, 1], tf.float32)
                #     distances_before = tf.zeros([1, 1], tf.float32)
                #
                # else:
                are_founds_before = _are_founds_before
                closest_labels_before = _closest_labels_before
                distances_before = _distances_before

                are_founds_after = _are_founds_after
                closest_labels_after = _closest_labels_after
                distances_after = _distances_after

            else:
                # if params.model['model_type'].lower() == 'conv':
                #     are_founds_before = tf.zeros([1, 1], tf.float32)
                #     closest_labels_before = tf.zeros([1, 1], tf.float32)
                #     distances_before = tf.zeros([1, 1], tf.float32)
                # else:
                are_founds_before = tf.concat([are_founds_before, are_founds_before], axis=1)
                closest_labels_before = tf.concat([closest_labels_before, closest_labels_before], axis=1)
                distances_before = tf.concat([distances_before, distances_before], axis=1)

                are_founds_after = tf.concat([are_founds_after, _are_founds_after], axis=1)
                closest_labels_after = tf.concat([closest_labels_after, _closest_labels_after], axis=1)
                distances_after = tf.concat([distances_after, _distances_after], axis=1)

    return avg_recall_after_model, normalized_recalls_after_model, distance_from_after_model_q_to_p, \
           avg_recall_before_model, normalized_recalls_before_model, distance_from_before_model_q_to_p, \
           delta_before_after_model, subset_labels, are_founds_after,closest_labels_after,distances_after,\
           are_founds_before,closest_labels_before,distances_before,

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

    global_step = tf.train.get_global_step()
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    baseline_source_embeddings = features['source_embeddings']
    # if params.model['model_type'].lower() == 'conv':
    #     baseline_source_embeddings = features['source_embeddings']
    # else:
    #     baseline_question_embeddings = features
    before_model_embeddings = tf.nn.l2_normalize(baseline_source_embeddings,
                                                     name='normalized_before_model_source_embeddings', axis=1)
    # -----------------------------------------------------------
    after_model_embeddings = orchestrate_model(features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        if params.executor["is_debug_mode"]:
            avg_recall_after_model, normalized_recalls_after_model, distance_from_after_model_q_to_p, \
            avg_recall_before_model,normalized_recalls_before_model, distance_from_before_model_q_to_p, \
            delta_before_after_model, actual_labels,are_founds_after,closest_labels_after,distances_after, \
            are_founds_before, closest_labels_before, distances_before   = recall_fn(base_data_path, params, before_model_embeddings, after_model_embeddings,KN_FILE_NAMES)

            results = {'embeddings': after_model_embeddings}
            results['avg_recall_after_model'] = avg_recall_after_model
            results['normalized_recalls_after_model'] = normalized_recalls_after_model
            results['distance_from_after_model_q_to_p'] = distance_from_after_model_q_to_p
            results['avg_recall_before_model'] = avg_recall_before_model
            results['normalized_recalls_before_model'] = normalized_recalls_before_model
            results['distance_from_before_model_q_to_p'] = distance_from_before_model_q_to_p
            results['delta_before_after_model'] = delta_before_after_model
            results['actual_labels'] = actual_labels
            results['are_founds_after'] = are_founds_after
            results['closest_labels_after'] = closest_labels_after
            results['distances_after'] = distances_after
            results['are_founds_before'] = are_founds_before
            results['closest_labels_before'] = closest_labels_before
            results['distances_before'] = distances_before

        else:
            results = after_model_embeddings

        return tf.estimator.EstimatorSpec(mode=mode, predictions=results)

    #if params.model['model_type'].lower() == 'conv':
    targets = labels['target_embeddings']
    labels = tf.cast(labels['target_labels'], tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    # else:
    #     # question_embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    #     paragraphs = labels[:, 0:params.files['pre_trained_files']['embedding_dim']]
    #     labels = labels[:, params.files['pre_trained_files']['embedding_dim']:params.files['pre_trained_files']['embedding_dim']+1]

    targets = tf.nn.l2_normalize(targets, name='normalized_target_embeddings', axis=1)
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
        #global_step_ = tf.Print(global_step, [global_step], message="Value of global step")
        avg_recall_after_model, normalized_recalls_after_model, distance_from_after_model_q_to_p, \
        avg_recall_before_model, normalized_recalls_before_model, distance_from_before_model_q_to_p, \
        delta_before_after_model, actual_labels, are_founds_after,closest_labels_after,distances_after, \
        are_founds_before, closest_labels_before, distances_before = recall_fn(base_data_path, params, before_model_embeddings, after_model_embeddings,KN_FILE_NAMES)
        #loss_over_recall_top_1 = tf.cast(loss, dtype=tf.float64) + tf.keras.backend.epsilon() / normalized_recalls_after_model[0]
        tf.summary.scalar("avg_recall", avg_recall_after_model)
        tf.summary.scalar("recall_top_1", normalized_recalls_after_model[0])
        tf.summary.scalar("recall_top_2", normalized_recalls_after_model[1])
        #tf.summary.scalar("loss/recall_top_1", loss_over_recall_top_1)
        eval_metric_ops= {"recall_top_1": tf.metrics.mean(normalized_recalls_after_model[0])}
        eval_metric_ops["recall_top_2"] = tf.metrics.mean(normalized_recalls_after_model[1])
        eval_metric_ops["avg_recall"] = tf.metrics.mean(avg_recall_after_model)
        #eval_metric_ops["loss/recall_top_1"] = tf.metrics.mean(loss_over_recall_top_1)
        return tf.estimator.EstimatorSpec(mode, loss= loss, eval_metric_ops=eval_metric_ops)


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