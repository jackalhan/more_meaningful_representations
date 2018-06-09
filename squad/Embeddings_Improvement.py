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
    question_to_ground_truth_distance, calculate_recalls, \
    load_embeddings, next_batch,get_question_and_paragraph_embeddings
from helper.quadratic_loss import euclidean_distance_loss
import numpy as np
from helper.models import model_1
import helper.parser as parser

def define_pre_executions(args, params):
    if args.analyze_labels:
        analysis = analyze_labes(args.labels_file)

    else:
        if args.split_train_test:
            file_paths = train_test_splitter(args.question_embeddings_file,
                                             args.paragraph_embeddings_file,
                                             args.labels_file,
                                             params.train_splitter_rate,
                                             params.eval_question_size_for_recall,
                                             args.limit_data)
            params.train_size = file_paths['train_question_size']
            params.eval_size = file_paths['eval_question_size']
            params.num_labels = file_paths['num_labels']
            params.save(json_path)
            print('Done with splitting')
            sys.exit()
        else:
            file_paths = {}
            file_paths['train_question_embeddings'] = args.train_question_embeddings_file
            file_paths['train_paragraph_embeddings'] = args.train_paragraph_embeddings_file
            file_paths['train_paragraph_labels'] = args.train_label_file

            file_paths['test_question_embeddings'] = args.test_question_embeddings_file
            file_paths['test_paragraph_embeddings'] = args.test_paragraph_embeddings_file
            file_paths['test_paragraph_labels'] = args.test_label_file

            file_paths['test_recall_question_embeddings'] = args.test_recall_question_embeddings
            file_paths['paragraph_embeddings'] = args.paragraph_embeddings_file
    return file_paths

def trace_metrics(eval_metric_ops, indx, is_add_as_collection=True, is_train=True, is_display=True):
    # to observe variable in trainable_variables collection
    variables_names = [v.name for v in tf.trainable_variables()]
    all_trainable_variables = sess.run(variables_names)
    for k, v in zip(variables_names, all_trainable_variables):
        if k == 'model/model/fc/_fc1/weights:0':
            mean_of_weights_in_trainable_variables = np.mean(v)

    # sort emo and add metrics for plots
    _emo = collections.OrderedDict(sorted(eval_metric_ops.items()))
    sub_metrics_for_plot = [indx]
    for key in _emo:
        if is_display:
            print("{}: {}".format(key, _emo[key]))
        if is_add_as_collection:
            sub_metrics_for_plot.extend([_emo[key]])
    if is_display:
        print("top_1_normalized_recall: {}".format(_emo['normalized_recalls'][0]))
        print("top_2_normalized_recall: {}".format(_emo['normalized_recalls'][1]))
        print("mean_of_weights_in_trainable_variable collection: {}".format(mean_of_weights_in_trainable_variables))

    if is_add_as_collection:
        sub_metrics_for_plot.extend([_emo['normalized_recalls'][0]])
        sub_metrics_for_plot.extend([_emo['normalized_recalls'][1]])
        sub_metrics_for_plot.extend([mean_of_weights_in_trainable_variables])

    if is_add_as_collection:
        if is_train:
            sub_metrics_for_plot.extend(['Train'])
        else:
            sub_metrics_for_plot.extend(['Test'])

    return sub_metrics_for_plot




if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.get_parser().parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # split the files or use the provided files
    file_paths = define_pre_executions(args, params)

    questions = tf.placeholder(tf.float32, [None, params.embedding_dim])
    labels = tf.placeholder(tf.int32, [None, 1])
    paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim])
    recall_paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim])

    # call the model
    with tf.variable_scope('model'):
        trained_question_embeddings = model_1(questions, params)

    # normalize paragraphs and some metric calculation
    with tf.name_scope('normalization_metrics') as scope:
        normalized_questions = tf.nn.l2_normalize(questions, name='ques', axis=1)
        normalized_paragraphs = tf.nn.l2_normalize(paragraphs, name='pars', axis=1)
        normalized_recall_paragraphs = tf.nn.l2_normalize(recall_paragraphs, name='recall_pars', axis=1)
        mean_of_norm_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_paragraphs), axis=1)))
        mean_of_norm_ques = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(trained_question_embeddings), axis=1)))
        mean_of_norm_recall_pars = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(normalized_recall_paragraphs), axis=1)))
        tf.summary.scalar("length_of_norm_pars", mean_of_norm_pars)
        tf.summary.scalar("length_of_norm_ques", mean_of_norm_ques)
        tf.summary.scalar("length_of_norm_recall_pars", mean_of_norm_recall_pars)

    with tf.name_scope('other_metrics') as scope:
        avg_recall, recalls, normalized_recalls, number_of_questions, q_index_and_cos = calculate_recalls(trained_question_embeddings,
                                                                                                          normalized_recall_paragraphs,
                                                                                                          labels,
                                                                                                          params)
        tf.summary.scalar("avg_recall", avg_recall)

        ground_truth_euclidean_distances = euclidean_distance_loss(normalized_questions, normalized_paragraphs,
                                                                   params=None, is_training=False, type='old_loss')

        closest_euclidean_distances = closest_distance(normalized_questions,
                                                       normalized_recall_paragraphs,
                                                       input_type='q2p',
                                                       score_type='euclidean')


    with tf.variable_scope('loss'):
        # Loss calculation
        if params.loss == 'abs_reg_loss':
            _loss = euclidean_distance_loss(trained_question_embeddings, normalized_paragraphs, params,type='new_loss')

        else:
            raise ValueError("Loss strategy not recognized: {}".format(params.loss))

        # Add Loss to Losses
        tf.losses.add_loss(_loss)
        loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', loss)

    with tf.variable_scope("metric_op"):
        eval_metric_ops = {"length_of_norm_pars": mean_of_norm_pars}
        eval_metric_ops['length_of_norm_ques'] =  mean_of_norm_ques
        eval_metric_ops['length_of_norm_recall_pars'] = mean_of_norm_recall_pars
        eval_metric_ops['avg_recall'] = avg_recall
        eval_metric_ops['recalls'] = recalls
        eval_metric_ops['normalized_recalls'] = normalized_recalls
        eval_metric_ops['question_set_size'] = tf.shape(trained_question_embeddings)[0]
        eval_metric_ops['paragraph_set_size'] = tf.shape(normalized_paragraphs)[0]
        eval_metric_ops['paragraph_recall_set_size'] = tf.shape(normalized_recall_paragraphs)[0]
        eval_metric_ops['loss'] = loss

    with tf.variable_scope('optimization'):

        if params.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        sess.run(init_op)

        #TRAINING DATA
        training_question_embeddings = load_embeddings(file_paths['train_question_embeddings'])
        training_labels = training_question_embeddings[:, params.embedding_dim:params.embedding_dim+1]
        training_question_embeddings = training_question_embeddings[:, :params.embedding_dim]
        training_paragraph_embeddings = load_embeddings(file_paths['train_paragraph_embeddings'])

        # TESTING DATA
        testing_question_embeddings, testing_paragraph_embeddings = get_question_and_paragraph_embeddings(False, file_paths['test_question_embeddings'],file_paths['test_paragraph_embeddings'], params)
        testing_labels = testing_question_embeddings[:, params.embedding_dim:params.embedding_dim + 1]
        testing_question_embeddings = testing_question_embeddings[:, :params.embedding_dim]
        all_paragraph_embeddings_for_recall = load_embeddings(args.paragraph_embeddings_file)

        # CREATING BATCHES
        all_metrics_for_plot = []
        total_batch = int(training_labels.shape[0]/ params.batch_size)

        print("=" * 50)
        print('::Testing values wthout training:')
        print("=" * 50)
        _emo = sess.run(
            eval_metric_ops, feed_dict={
                questions: testing_question_embeddings,
                paragraphs: testing_paragraph_embeddings,
                labels: testing_labels,
                recall_paragraphs: all_paragraph_embeddings_for_recall,
            })

        all_metrics_for_plot.append(trace_metrics(_emo, 0, True, False, True))


        for epoch in range(params.num_epochs):
            print("*" * 50)
            print("Epoch:", (epoch + 1), "is started with a total batch of {}".format(total_batch))
            print("Each batch size is {}".format(params.batch_size))
            avg_loss_value = 0
            for i in range(total_batch):
                #  ... without sampling from Python and without a feed_dict !
                batch_training_question_embeddings, batch_training_labels, batch_training_paragraph_embeddings = \
                    next_batch(i*params.batch_size, params.batch_size,
                                 training_question_embeddings,
                                 training_labels,
                                 training_paragraph_embeddings)

                # execute the training by feeding batches

                _, loss_value, new_question_embeddings, _emo= sess.run([train_op, loss, trained_question_embeddings, eval_metric_ops], feed_dict={
                                                questions: batch_training_question_embeddings,
                                                paragraphs: batch_training_paragraph_embeddings,
                                                labels: batch_training_labels,
                                                recall_paragraphs:all_paragraph_embeddings_for_recall,
                                            })

                all_metrics_for_plot.append(trace_metrics(_emo, epoch*i, True, True, False))
                # We regularly check the loss in each 100 iterations
                if i % 100 == 0:
                    print(25 * '-')
                    print('::Training values at iter {}::'.format(i) )
                    _ = trace_metrics(_emo, epoch * i, False, True, True)
                avg_loss_value += loss_value / total_batch

            print("=" * 50)
            print('::Testing values:')
            print("=" * 50)
            _emo = sess.run(
                eval_metric_ops, feed_dict={
                    questions: testing_question_embeddings,
                    paragraphs: testing_paragraph_embeddings,
                    labels: testing_labels,
                    recall_paragraphs: all_paragraph_embeddings_for_recall,
                })

            all_metrics_for_plot.append(trace_metrics(_emo,(epoch+1) * total_batch, True, False, True))

            print("Epoch:", (epoch + 1), "avg loss =", "{:.3f}".format(avg_loss_value))
            print("*" * 50)

        # raw_ques = ds._load_embeddings(args.pretrained_embedding_file)
        # new_question_embeddings = sess.run(out, feed_dict={
        #     questions: raw_ques
        # })
        #
        # print('New Embeddings are predicted')
        # print('{}'.format(raw_ques.shape))
        _e = args.question_embeddings_file.rpartition(os.path.sep)
        path_e = _e[0]

        if args.is_prediction:
            print('Question embedding are getting prediceted')
            non_trained_all_questions = load_embeddings(args.question_embeddings_file)
            trained_all_question_embeddings = sess.run(trained_question_embeddings, feed_dict={
                questions: non_trained_all_questions
            })


        if args.is_run_metrics:
            print('Metrics are getting saved')
            file_name = os.path.join(path_e, 'epoch_' + str(params.num_epochs) + '_margin_' + str(params.margin) + '_lr_' + str(params.learning_rate) +
                                     '_scaling_factor_' + str(params.scaling_factor) + '_l2_reg_' + str(params.l2_regularizer) + '.csv')
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

            print(50 * '-')
            print('Calculating the distances differences for All Dataset')
            print(50 * '-')
            print('Before Trained Model')
            print('[question to ground truth paragraph distance] - [question to closest paragraph distance]')
            non_trained_all_questions = load_embeddings(args.question_embeddings_file)
            all_paragraphs = np.squeeze(load_embeddings(args.paragraph_embeddings_file))
            all_labels = pd.read_csv(args.labels_file)
            all_mapped_non_trained_qs_to_ps  = all_paragraphs[all_labels['v'].values]

            non_trained_ground_truth_distances = question_to_ground_truth_distance(non_trained_all_questions,
                                                                                   all_mapped_non_trained_qs_to_ps,
                                                                                    params.eval_question_size_for_recall,
                                                                                    sess,
                                                                                    ground_truth_euclidean_distances,
                                                                                    questions,
                                                                                    paragraphs)

            non_trained_closest_distances = question_to_closest_distance(non_trained_all_questions,
                                                                          all_paragraphs,
                                                                          params.eval_question_size_for_recall,
                                                                          sess,
                                                                          closest_euclidean_distances,
                                                                          questions,
                                                                          recall_paragraphs)


            non_trained_differences = non_trained_ground_truth_distances - non_trained_closest_distances

            np.savetxt(os.path.join(path_e, 'non_trained_distances_epoch_' + str(params.num_epochs) + '_margin_' + str(params.margin) + '_lr_' + str(params.learning_rate) +
                        '_scaling_factor_' + str(params.scaling_factor) + '_l2_reg_' + str(params.l2_regularizer) + '.csv')
                , non_trained_differences, delimiter=",")

            print(50 * '-')
            print('After Trained Model')
            print(
                '[improved question to ground truth paragraph distance] - [improved question to closest paragraph distance]')
            trained_ground_truth_distances = question_to_ground_truth_distance(trained_all_question_embeddings,
                                                                                   all_mapped_non_trained_qs_to_ps,
                                                                                   params.eval_question_size_for_recall,
                                                                                   sess,
                                                                                   ground_truth_euclidean_distances,
                                                                                   questions,
                                                                                   paragraphs)

            trained_closest_distances = question_to_closest_distance(trained_all_question_embeddings,
                                                                         all_paragraphs,
                                                                         params.eval_question_size_for_recall,
                                                                         sess,
                                                                         closest_euclidean_distances,
                                                                         questions,
                                                                         recall_paragraphs)

            trained_differences = trained_ground_truth_distances - trained_closest_distances

            np.savetxt(os.path.join(path_e, 'trained_distances_epoch_' + str(params.num_epochs) + '_margin_' + str(
                params.margin) + '_lr_' + str(params.learning_rate) +
                                    '_scaling_factor_' + str(params.scaling_factor) + '_l2_reg_' + str(
                params.l2_regularizer) + '.csv')
                       , trained_differences, delimiter=",")



    #print("".format(len(all_metrics_for_plot)))
    print('Done')

