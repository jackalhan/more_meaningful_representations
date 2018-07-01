"""Train the model"""


import os
import math
import pandas as pd
import tensorflow as tf
import sys
import collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import Params,\
    closest_distance, question_to_closest_distance, dump_embeddings, \
    question_to_ground_truth_distance, question_to_random_paragraph_distance, calculate_recalls, \
    load_embeddings, next_batch,get_question_and_paragraph_embeddings, define_pre_executions
from helper.loss import euclidean_distance_loss
import numpy as np
from helper.models import orchestrate_model
import helper.parser as parser
import random


def trace_metrics(eval_metric_ops, indx, is_add_as_collection=True, is_train=True, is_display=True):
    # to observe variable in trainable_variables collection
    variables_names = [v.name for v in tf.trainable_variables()]
    all_trainable_variables = sess.run(variables_names)
    mean_of_weights_in_trainable_variables = 0
    for k, v in zip(variables_names, all_trainable_variables):
        mean_of_weights_in_trainable_variables += np.mean(v)

    # sort emo and add metrics for plots
    _emo = collections.OrderedDict(sorted(eval_metric_ops.items()))
    sub_metrics_for_plot = [indx]
    for key in _emo:
        if is_display:
            print("{}: {}".format(key, _emo[key]))
        if is_add_as_collection:
            sub_metrics_for_plot.extend([_emo[key]])
    if is_display:
        #print("top_1_normalized_recall: {}".format(_emo['normalized_recalls'][0]))
        #print("top_2_normalized_recall: {}".format(_emo['normalized_recalls'][1]))
        print("mean_of_weights_in_trainable_variable collection: {}".format(mean_of_weights_in_trainable_variables))

    if is_add_as_collection:
        #sub_metrics_for_plot.extend([_emo['normalized_recalls'][0]])
        #sub_metrics_for_plot.extend([_emo['normalized_recalls'][1]])
        sub_metrics_for_plot.extend([mean_of_weights_in_trainable_variables])

    if is_add_as_collection:
        if is_train:
            sub_metrics_for_plot.extend(['Train'])
        else:
            sub_metrics_for_plot.extend(['Test'])

    return sub_metrics_for_plot

def measure_distances(question_embeddings,
                      paragraph_embeddings,
                      recall_paragraph_embeddings,
                      sess,
                      closest_distance_op,
                      groundtruth_distance_op,
                      question_tensor,
                      paragraph_tensor,
                      paragraph_recall_tensor,
                      params,
                      save_path=None,
                      from_dist="Q_to_GroundTruth_P",
                      to_dist="Q_to_Closest_P",
                      log_step='Before_Trained_Model'):
    print(50 * '-')
    print('Calculating the distance differences between question and paragraphs')
    print(50 * '-')
    print(log_step)
    print(from_dist, 'and', to_dist)


    _ground_truth_distances = question_to_ground_truth_distance(question_embeddings,
                                                               paragraph_embeddings,
                                                               params.eval_question_size_for_recall,
                                                               sess,
                                                               groundtruth_distance_op,
                                                               question_tensor,
                                                               paragraph_tensor)

    _random_distances = question_to_random_paragraph_distance(question_embeddings,
                                                                paragraph_embeddings,
                                                                params.eval_question_size_for_recall,
                                                                sess,
                                                                groundtruth_distance_op,
                                                                question_tensor,
                                                                paragraph_tensor)


    _closest_distances = question_to_closest_distance(question_embeddings,
                                                      recall_paragraph_embeddings,
                                                      params.eval_question_size_for_recall,
                                                      sess,
                                                      closest_distance_op,
                                                      question_tensor,
                                                      paragraph_recall_tensor)

    _distance_differences = _ground_truth_distances - _closest_distances

    if save_path is not None:
        np.savetxt(os.path.join(save_path,"{}_{}_epoch_{}_margin_{}_lr_{}_scaling_factor_{}_l2_reg_{}.csv".format(params.model_name,
                                                                                                                  from_dist,
                                                                                                            params.num_epochs,
                                                                                                            params.margin,
                                                                                                            params.learning_rate,
                                                                                                            params.scaling_factor,
                                                                                                            params.l2_regularizer))
                   , _ground_truth_distances, delimiter=",")

        np.savetxt(
            os.path.join(save_path, "{}_{}_epoch_{}_margin_{}_lr_{}_scaling_factor_{}_l2_reg_{}.csv".format(params.model_name,
                                                                                                            'Q_to_random_P',
                                                                                                            params.num_epochs,
                                                                                                            params.margin,
                                                                                                            params.learning_rate,
                                                                                                            params.scaling_factor,
                                                                                                            params.l2_regularizer))
            , _random_distances, delimiter=",")

        np.savetxt(
            os.path.join(save_path, "{}_{}_epoch_{}_margin_{}_lr_{}_scaling_factor_{}_l2_reg_{}.csv".format(params.model_name,
                                                                                                            to_dist,
                                                                                                            params.num_epochs,
                                                                                                            params.margin,
                                                                                                            params.learning_rate,
                                                                                                            params.scaling_factor,
                                                                                                            params.l2_regularizer))
            , _closest_distances, delimiter=",")
        np.savetxt(
            os.path.join(save_path, "{}_{}_{}_epoch_{}_margin_{}_lr_{}_scaling_factor_{}_l2_reg_{}.csv".format(params.model_name,
                                                                                                               from_dist,
                                                                                                            to_dist,
                                                                                                            params.num_epochs,
                                                                                                            params.margin,
                                                                                                            params.learning_rate,
                                                                                                            params.scaling_factor,
                                                                                                            params.l2_regularizer))
            , _distance_differences, delimiter=",")

    return _ground_truth_distances, _closest_distances, _random_distances, _distance_differences


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parser.get_parser().parse_args()
    assert os.path.isfile(args.json_path), "No json configuration file found at {}".format(args.json_path)
    params = Params(args.json_path)

    # split the files or use the provided files
    params = define_pre_executions(params, args.json_path)
    base_data_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])

    tf.logging.info("Creating the model...")
    questions = tf.placeholder(tf.float32, [None, params.files['pre_trained_files']['embedding_dim']], name='input_questions')
    labels = tf.placeholder(tf.int32, [None, 1], name='input_labels')
    paragraphs = tf.placeholder(tf.float32, [None, params.files['pre_trained_files']['embedding_dim']], name='input_paragraphs')
    recall_paragraphs = tf.placeholder(tf.float32, [None, params.files['pre_trained_files']['embedding_dim']], name='input_recall_paragraphs')

    # call the model
    trained_question_embeddings = orchestrate_model(questions, params)

    normalized_questions = tf.nn.l2_normalize(questions, name='normalize_questions', axis=1)
    normalized_paragraphs = tf.nn.l2_normalize(paragraphs, name='normalize_paragraphs', axis=1)
    normalized_recall_paragraphs = tf.nn.l2_normalize(recall_paragraphs, name='normalize_recall_pars', axis=1)
    mean_of_norm_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_paragraphs), axis=1)))
    mean_of_norm_ques = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(trained_question_embeddings), axis=1)))
    mean_of_norm_recall_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_recall_paragraphs), axis=1)))
    tf.summary.scalar("question_embedding_mean_norm", mean_of_norm_pars)
    tf.summary.scalar("paragraph_embedding_mean_norm", mean_of_norm_ques)
    tf.summary.scalar("all_paragraphs_norm", mean_of_norm_recall_pars)

    recalls, normalized_recalls = calculate_recalls(trained_question_embeddings,normalized_recall_paragraphs, labels, params)

    avg_recall = tf.reduce_mean(normalized_recalls)



    tf.summary.scalar("avg_recall", avg_recall)
    tf.summary.scalar("recall_top_1", normalized_recalls[0])
    tf.summary.scalar("recall_top_2", normalized_recalls[1])


    # ground_truth_euclidean_distances = euclidean_distance_loss(normalized_questions, normalized_paragraphs, params, is_training=False)
    #
    # closest_euclidean_distances = closest_distance(normalized_questions,
    #                                                normalized_recall_paragraphs,
    #                                                input_type='q2p',
    #                                                score_type='euclidean')



    # Loss calculation
    if params.loss['name'] == 'abs_reg_loss':
        _loss = euclidean_distance_loss(trained_question_embeddings, normalized_paragraphs, params)

    else:
        raise ValueError("Loss strategy not recognized: {}".format(params.loss['name']))

    # Add Loss to Losses

    # 1st alternative
    # -------------------------------------------------------
    # l2_loss = tf.losses.get_regularization_loss()
    # _loss += l2_loss
    # loss = _loss
    # tf.summary.scalar('final_loss', loss)

    # 2nd alternative
    # -------------------------------------------------------
    tf.losses.add_loss(_loss)
    loss = tf.losses.get_total_loss()
    loss_over_recall_top_1 = tf.cast(loss, dtype=tf.float64) + tf.keras.backend.epsilon() / normalized_recalls[0]
    tf.summary.scalar("loss/recall_top_1", loss_over_recall_top_1)
    tf.summary.scalar('loss', loss)

    eval_metric_ops = {"paragraph_embedding_mean_norm": mean_of_norm_pars}
    eval_metric_ops['question_embedding_mean_norm'] =  mean_of_norm_ques
    eval_metric_ops['all_paragraphs_norm'] = mean_of_norm_recall_pars
    eval_metric_ops['avg_recall'] = avg_recall
    # eval_metric_ops['recalls'] = recalls
    # eval_metric_ops['normalized_recalls'] = normalized_recalls
    # eval_metric_ops['question_set_size'] = tf.shape(trained_question_embeddings)[0]
    # eval_metric_ops['paragraph_set_size'] = tf.shape(normalized_paragraphs)[0]
    # eval_metric_ops['paragraph_recall_set_size'] = tf.shape(normalized_recall_paragraphs)[0]
    eval_metric_ops["recall_top_1"]= normalized_recalls[0]
    eval_metric_ops["recall_top_2"]= normalized_recalls[1]
    eval_metric_ops['loss'] = loss
    eval_metric_ops["loss/recall_top_1"] = loss_over_recall_top_1

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # ALTERNATIVE OPTIMIZATION 1
    # ---------------------------------------------
    # train_op = tf.contrib.layers.optimize_loss(
    #     loss,
    #     global_step,
    #     params.learning_rate,
    #     params.optimizer
    # )

    # ALTERNATIVE OPTIMIZATION 2
    # ---------------------------------------------
    # loss_params = tf.trainable_variables()
    # gradients = tf.gradients(loss, loss_params, name='gradients')
    #
    # if params.optimizer == 'Adam':
    #     optimizer = tf.train.AdamOptimizer(params.learning_rate)
    # else:
    #     optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    #
    # update = optimizer.apply_gradients(zip(gradients, loss_params))
    # with tf.control_dependencies([update]):
    #     train_op = tf.no_op(name='train_op')

    # ALTERNATIVE OPTIMIZATION 3
    # ---------------------------------------------
    if params.optimizer['name'] == 'Adam':
        optimizer = tf.train.AdamOptimizer(params.optimizer['learning_rate'])
    else:
        optimizer = tf.train.GradientDescentOptimizer(params.optimizer['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=global_step)

    # # Save the grads with tf.summary.histogram:
    # for index, grad in enumerate(grads):
    #     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
    #
    init_op = tf.global_variables_initializer()

    # Merge all summary inforation.
    summary_op = tf.summary.merge_all()
    summaries_dir = os.path.join(params.executor["model_dir"], "low_level_log", 'non_est_' + params.model["active_model"])
    # start the session
    with tf.Session() as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test')

        #TRAINING DATA
        training_question_embeddings = load_embeddings(os.path.join(base_data_path, params.files['train_loss']['question_embeddings']))
        training_paragraph_embeddings = load_embeddings(os.path.join(base_data_path, params.files['train_loss']['paragraph_embeddings']))
        training_labels = load_embeddings(
            os.path.join(base_data_path, params.files['train_loss']['question_labels']))
        training_labels = np.reshape(training_labels, [-1, 1])
        # TESTING DATA
        testing_question_embeddings = load_embeddings(os.path.join(base_data_path, params.files['test_subset_loss']['question_embeddings']))
        testing_paragraph_embeddings = load_embeddings(os.path.join(base_data_path, params.files['test_subset_loss']['paragraph_embeddings']))
        all_paragraph_embeddings = load_embeddings(os.path.join(base_data_path, params.files['test_subset_recall']['all_paragraph_embeddings']))
        testing_labels = load_embeddings(os.path.join(base_data_path, params.files['test_subset_recall']['question_labels']))
        testing_labels = np.reshape(testing_labels, [-1,1])

        # CREATING BATCHES
        all_metrics_for_plot = []
        total_batch = math.ceil(training_question_embeddings.shape[0]/ params.model['batch_size'])

        print("=" * 50)
        print('::Testing values without training:')
        print("=" * 50)

        _emo = sess.run(
            eval_metric_ops, feed_dict={
                questions: testing_question_embeddings,
                paragraphs: testing_paragraph_embeddings,
                labels: testing_labels,
                recall_paragraphs: all_paragraph_embeddings,
            })

        all_metrics_for_plot.append(trace_metrics(_emo, 0, True, False, True))
        tf.set_random_seed(params.model['seed'])
        counter = 0
        for epoch in range(1, params.model['num_epochs']+1):
            print("*" * 50)
            print("Epoch:", (epoch), "is started with a total batch of {}".format(total_batch))
            print("Each batch size is {}".format(params.model['batch_size']))

            training = list(zip(training_question_embeddings, training_labels,  training_paragraph_embeddings))
            #random.shuffle(training)
            training_question_embeddings, training_labels, training_paragraph_embeddings = zip(*training)
            training_question_embeddings, training_labels, training_paragraph_embeddings = np.asarray(training_question_embeddings), np.asarray(training_labels), np.asarray(training_paragraph_embeddings)
            avg_loss_value = 0
            for i in range(1, total_batch+1):
                counter += 1
                #  ... without sampling from Python and without a feed_dict !
                batch_training_question_embeddings, batch_training_labels, batch_training_paragraph_embeddings = \
                    next_batch((i-1)*params.model['batch_size'], params.model['batch_size'],
                                 training_question_embeddings,
                                 training_paragraph_embeddings,
                                 training_labels)

                summary,_, loss_value, new_question_embeddings, _emo= sess.run([summary_op,
                                                                                train_op,
                                                                                loss,
                                                                                trained_question_embeddings,
                                                                                eval_metric_ops],
                                                                               feed_dict={
                                                                                questions: batch_training_question_embeddings,
                                                                                paragraphs: batch_training_paragraph_embeddings,
                                                                                labels: batch_training_labels,
                                                                                recall_paragraphs:all_paragraph_embeddings,
                                                                            })
                train_writer.add_summary(summary, counter)
                train_writer.flush()
                all_metrics_for_plot.append(trace_metrics(_emo, counter, True, True, False))
                avg_loss_value += loss_value / total_batch

                # We regularly check the loss in each 100 iterations
                #if i % 100 == 0:

            print(25 * '-')
            # execute the training by feeding batches

            print('::Testing values at iter {}::'.format(counter))
            summary, _emo, new_question_embeddings = sess.run([summary_op, eval_metric_ops, trained_question_embeddings], feed_dict={
                questions: testing_question_embeddings,
                paragraphs: testing_paragraph_embeddings,
                labels: testing_labels,
                recall_paragraphs: all_paragraph_embeddings,
            })
            test_writer.add_summary(summary, counter)
            train_writer.flush()
            all_metrics_for_plot.append(trace_metrics(_emo, counter, True, False, True))
            no, av = sess.run(
                [normalized_recalls, avg_recall], feed_dict={
                    questions: new_question_embeddings,
                    paragraphs: testing_paragraph_embeddings,
                    labels: testing_labels,
                    recall_paragraphs: all_paragraph_embeddings,
                })

            print("norm_recall", no)
            print("av", av)
            print("Epoch:", epoch, "avg loss =", "{:.3f}".format(avg_loss_value))
            print("*" * 50)

    print('Done')

