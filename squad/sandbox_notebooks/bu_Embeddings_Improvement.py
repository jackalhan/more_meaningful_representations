"""Train the model"""


import os
import math
import pandas as pd
import tensorflow as tf
import sys
import collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import Params, train_test_splitter, analyze_labes, \
    closest_distance, question_to_closest_distance, dump_embeddings, \
    question_to_ground_truth_distance, question_to_random_paragraph_distance, calculate_recalls, \
    load_embeddings, next_batch,get_question_and_paragraph_embeddings, define_pre_executions
from helper.quadratic_loss import euclidean_distance_loss
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

    # Load the parameters from json file
    args = parser.get_parser().parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # split the files or use the provided files
    file_paths = define_pre_executions(args, params,json_path)

    questions = tf.placeholder(tf.float32, [None, params.embedding_dim], name='input_questions')
    labels = tf.placeholder(tf.int32, [None, 1], name='input_labels')
    paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim], name='input_paragraphs')
    recall_paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim], name='input_recall_paragraphs')

    # call the model
    trained_question_embeddings = orchestrate_model(questions, params)

    normalized_questions = tf.nn.l2_normalize(questions, name='normalize_questions', axis=1)
    normalized_paragraphs = tf.nn.l2_normalize(paragraphs, name='normalize_paragraphs', axis=1)
    normalized_recall_paragraphs = tf.nn.l2_normalize(recall_paragraphs, name='normalize_recall_pars', axis=1)
    mean_of_norm_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_paragraphs), axis=1)))
    mean_of_norm_ques = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(trained_question_embeddings), axis=1)))
    mean_of_norm_recall_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_recall_paragraphs), axis=1)))
    tf.summary.scalar("length_of_norm_pars", mean_of_norm_pars)
    tf.summary.scalar("length_of_norm_ques", mean_of_norm_ques)
    tf.summary.scalar("length_of_norm_recall_pars", mean_of_norm_recall_pars)

    avg_recall, recalls, normalized_recalls, number_of_questions, q_index_and_cos = calculate_recalls(trained_question_embeddings,
                                                                                                      normalized_recall_paragraphs,
                                                                                                      labels,
                                                                                                      params)
    tf.summary.scalar("avg_recall", avg_recall)
    tf.summary.scalar("top_1_normalized_recall", normalized_recalls[0])
    tf.summary.scalar("top_2_normalized_recall", normalized_recalls[1])
    ground_truth_euclidean_distances = euclidean_distance_loss(normalized_questions, normalized_paragraphs,
                                                               params=None,
                                                               is_training=False,
                                                               type='old_loss')

    closest_euclidean_distances = closest_distance(normalized_questions,
                                                   normalized_recall_paragraphs,
                                                   input_type='q2p',
                                                   score_type='euclidean')



    # Loss calculation
    if params.loss == 'abs_reg_loss':
        _loss = euclidean_distance_loss(trained_question_embeddings, normalized_paragraphs, params,type=params.loss_type)

    else:
        raise ValueError("Loss strategy not recognized: {}".format(params.loss))

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
    tf.summary.scalar('final_loss', loss)


    eval_metric_ops = {"length_of_norm_pars": mean_of_norm_pars}
    eval_metric_ops['length_of_norm_ques'] =  mean_of_norm_ques
    eval_metric_ops['length_of_norm_recall_pars'] = mean_of_norm_recall_pars
    eval_metric_ops['avg_recall'] = avg_recall
    eval_metric_ops['recalls'] = recalls
    eval_metric_ops['normalized_recalls'] = normalized_recalls
    eval_metric_ops['question_set_size'] = tf.shape(trained_question_embeddings)[0]
    eval_metric_ops['paragraph_set_size'] = tf.shape(normalized_paragraphs)[0]
    eval_metric_ops['paragraph_recall_set_size'] = tf.shape(normalized_recall_paragraphs)[0]
    eval_metric_ops["top_1_normalized_recall"]= normalized_recalls[0]
    eval_metric_ops["top_2_normalized_recall"]= normalized_recalls[1]
    eval_metric_ops['loss'] = loss

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
    if params.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # # Save the grads with tf.summary.histogram:
    # for index, grad in enumerate(grads):
    #     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
    #
    init_op = tf.global_variables_initializer()

    # Merge all summary inforation.
    summary_op = tf.summary.merge_all()
    summaries_dir = os.path.join(params.log_path, params.model_name, )
    # start the session
    with tf.Session() as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test')

        #TRAINING DATA
        training_question_embeddings = load_embeddings(file_paths['train_question_embeddings'])
        training_labels = training_question_embeddings[:, params.embedding_dim:params.embedding_dim+1]
        training_question_embeddings = training_question_embeddings[:, :params.embedding_dim]
        training_paragraph_embeddings = load_embeddings(file_paths['train_paragraph_embeddings'])

        # TESTING DATA
        testing_question_embeddings, testing_paragraph_embeddings = get_question_and_paragraph_embeddings(False, False, file_paths['test_question_embeddings'],file_paths['test_paragraph_embeddings'], params)
        testing_labels = testing_question_embeddings[:, params.embedding_dim:params.embedding_dim + 1]
        testing_question_embeddings = testing_question_embeddings[:, :params.embedding_dim]
        all_paragraph_embeddings_for_recall = load_embeddings(args.paragraph_embeddings_file)

        # CREATING BATCHES
        all_metrics_for_plot = []
        total_batch = math.ceil(training_labels.shape[0]/ params.batch_size)

        print("=" * 50)
        print('::Testing values wthout training:')
        print("=" * 50)
        # tqe = sess.run(trained_question_embeddings, feed_dict={
        #     questions: testing_question_embeddings,
        #     paragraphs: testing_paragraph_embeddings,
        #     labels: testing_labels,
        #     recall_paragraphs: all_paragraph_embeddings_for_recall,
        # })
        _emo = sess.run(
            eval_metric_ops, feed_dict={
                questions: testing_question_embeddings,
                paragraphs: testing_paragraph_embeddings,
                labels: testing_labels,
                recall_paragraphs: all_paragraph_embeddings_for_recall,
            })

        all_metrics_for_plot.append(trace_metrics(_emo, 0, True, False, True))
        tf.set_random_seed(params.seed)
        counter = 0
        for epoch in range(1, params.num_epochs+1):
            print("*" * 50)
            print("Epoch:", (epoch), "is started with a total batch of {}".format(total_batch))
            print("Each batch size is {}".format(params.batch_size))

            training = list(zip(training_question_embeddings, training_labels, training_paragraph_embeddings))
            #random.seed(params.seed)
            random.shuffle(training)
            training_question_embeddings, training_labels, training_paragraph_embeddings = zip(*training)
            training_question_embeddings, training_labels, training_paragraph_embeddings = np.asarray(training_question_embeddings), np.asarray(training_labels), np.asarray(training_paragraph_embeddings)
            avg_loss_value = 0
            for i in range(1, total_batch):
                counter += 1
                #  ... without sampling from Python and without a feed_dict !
                batch_training_question_embeddings, batch_training_labels, batch_training_paragraph_embeddings = \
                    next_batch((i-1)*params.batch_size, params.batch_size,
                                 training_question_embeddings,
                                 training_labels,
                                 training_paragraph_embeddings)

                summary,_, loss_value, new_question_embeddings, _emo= sess.run([summary_op, train_op, loss, trained_question_embeddings, eval_metric_ops], feed_dict={
                                                questions: batch_training_question_embeddings,
                                                paragraphs: batch_training_paragraph_embeddings,
                                                labels: batch_training_labels,
                                                recall_paragraphs:all_paragraph_embeddings_for_recall,
                                            })
                train_writer.add_summary(summary, counter)
                train_writer.flush()
                all_metrics_for_plot.append(trace_metrics(_emo, counter, True, True, False))
                avg_loss_value += loss_value / total_batch

                # We regularly check the loss in each 100 iterations
                if i % params.eval_step == 0:
                    print(25 * '-')
                    # print('::Training values at iter {}::'.format((epoch * i)+1) )
                    # _ = trace_metrics(_emo, epoch * i, False, True, False)

                    # execute the training by feeding batches

                    print('::Testing values at iter {}::'.format(counter))
                    summary, _emo = sess.run([summary_op, eval_metric_ops], feed_dict={
                            questions: testing_question_embeddings,
                            paragraphs: testing_paragraph_embeddings,
                            labels: testing_labels,
                            recall_paragraphs: all_paragraph_embeddings_for_recall,
                        })
                    test_writer.add_summary(summary, counter)
                    train_writer.flush()
                    all_metrics_for_plot.append(trace_metrics(_emo,counter, True, False, True))

            # _emo = sess.run( eval_metric_ops, feed_dict={
            #         questions: testing_question_embeddings,
            #         paragraphs: testing_paragraph_embeddings,
            #         labels: testing_labels,
            #         recall_paragraphs: all_paragraph_embeddings_for_recall,
            #     })
            #
            # all_metrics_for_plot.append(trace_metrics(_emo,counter, True, False, False))

            print("Epoch:", epoch, "avg loss =", "{:.3f}".format(avg_loss_value))
            print("*" * 50)

        _e = args.question_embeddings_file.rpartition(os.path.sep)
        path_e = _e[0]

        if args.is_prediction:
            print('Question embedding are getting prediceted')
            non_trained_all_questions = load_embeddings(args.question_embeddings_file)
            trained_all_question_embeddings = sess.run(trained_question_embeddings, feed_dict={
                questions: non_trained_all_questions
            })


        if args.is_run_metrics:
            print('Metrics.....')
            file_name = os.path.join(path_e,"{}_epoch_{}_margin_{}_lr_{}_scaling_factor_{}_l2_reg_{}.csv".format(params.model_name,
                                                                                                                 params.num_epochs,
                                                                                                            params.margin,
                                                                                                            params.learning_rate,
                                                                                                            params.scaling_factor,
                                                                                                            params.l2_regularizer))
            df_metrics = pd.DataFrame(data=all_metrics_for_plot, columns=['indx',
                                                                          'avg_recall',
                                                                          'length_of_norm_pars',
                                                                          'length_of_norm_ques',
                                                                          'length_of_norm_recall_pars',
                                                                          'loss',
                                                                          'normalized_recalls',
                                                                          'paragraph_recall_set_size',
                                                                          'paragraph_set_size',
                                                                          'question_set_size',
                                                                          'recalls',
                                                                          'top_1_normalized_recall',
                                                                          'top_2_normalized_recall',
                                                                          'mean_of_weights_in_trainable_variable',
                                                                          'mode'])
            df_metrics.to_csv(file_name)

            # all_paragraphs = np.squeeze(load_embeddings(args.paragraph_embeddings_file))
            # all_labels = pd.read_csv(args.labels_file)
            # all_mapped_non_trained_qs_to_ps = all_paragraphs[all_labels['v'].values]
            #
            # non_trained_differences = \
            #     measure_distances(question_embeddings=non_trained_all_questions,
            #                   paragraph_embeddings=all_mapped_non_trained_qs_to_ps,
            #                   recall_paragraph_embeddings=all_paragraphs,
            #                   sess=sess,
            #                   closest_distance_op=closest_euclidean_distances,
            #                   groundtruth_distance_op=ground_truth_euclidean_distances,
            #                   question_tensor=questions,
            #                   paragraph_tensor=paragraphs,
            #                   paragraph_recall_tensor=recall_paragraphs,
            #                   params=params,
            #                   save_path=path_e,
            #                   from_dist="Q_to_GroundTruth_P",
            #                   to_dist="Q_to_Closest_P",
            #                   log_step='Before_Trained_Model'
            #                   )
            # trained_differences = \
            #     measure_distances(question_embeddings=trained_all_question_embeddings,
            #                       paragraph_embeddings=all_mapped_non_trained_qs_to_ps,
            #                       recall_paragraph_embeddings=all_paragraphs,
            #                       sess=sess,
            #                       closest_distance_op=closest_euclidean_distances,
            #                       groundtruth_distance_op=ground_truth_euclidean_distances,
            #                       question_tensor=questions,
            #                       paragraph_tensor=paragraphs,
            #                       paragraph_recall_tensor=recall_paragraphs,
            #                       params=params,
            #                       save_path=path_e,
            #                       from_dist="Improved_Q_to_GroundTruth_P",
            #                       to_dist="Improved_Q_to_Closest_P",
            #                       log_step='After_Trained_Model'
            #                       )
            # non_and_trained_differences = \
            #     measure_distances(question_embeddings=trained_all_question_embeddings,
            #                       paragraph_embeddings=all_mapped_non_trained_qs_to_ps,
            #                       recall_paragraph_embeddings=all_paragraphs,
            #                       sess=sess,
            #                       closest_distance_op=closest_euclidean_distances,
            #                       groundtruth_distance_op=ground_truth_euclidean_distances,
            #                       question_tensor=questions,
            #                       paragraph_tensor=paragraphs,
            #                       paragraph_recall_tensor=recall_paragraphs,
            #                       params=params,
            #                       save_path=path_e,
            #                       from_dist="Q_to_GroundTruth_P",
            #                       to_dist="Improved_Q_to_Closest_P",
            #                       log_step='After_Trained_Model'
            #                       )

    print('Done')

