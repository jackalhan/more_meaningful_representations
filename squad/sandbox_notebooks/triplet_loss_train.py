"""Train the model"""

import argparse
import os

import tensorflow as tf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from triplet_loss.input_fn import train_input_fn
from triplet_loss.input_fn import test_input_fn
from triplet_loss.input_fn import live_input_fn, test_recall_input_fn
from triplet_loss.model_fn import model_fn
from triplet_loss.utils import Params, train_test_splitter, dump_embeddings, analyze_labes
import numpy as np
import triplet_loss.my_dataset as ds

parser = argparse.ArgumentParser()

# model path, the pretrained embeddings for questions, paragraphs and their mappings label file
parser.add_argument('--model_dir',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function',
                    help="Experiment directory containing params.json")
parser.add_argument('--question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
                    help="qustion embeddings_file")
parser.add_argument('--paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_paragraph_embeddings.hdf5',
                    help="paragraph embeddings_file")
parser.add_argument('--labels_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_q_to_p_mappings.csv',
                    help="labels_file")

# run configrations
parser.add_argument('--is_train',
                    default=True,
                    help="Run for the training")
parser.add_argument('--is_test',
                    default=True,
                    help="Run for the testing")
parser.add_argument('--is_prediction_for_evaluation',
                    default=True,
                    help="Run eval for the prediction")
parser.add_argument('--is_recall_comparision_with_baseline',
                    default=True,
                    help="Recall comparision with baseline")

parser.add_argument('--is_prediction_for_live',
                    default=True,
                    help="Run live for the prediction")
parser.add_argument('--is_dump_predictions',
                    default=True,
                    help="whether dump the prediction or not")

# data train/eval split configrations
parser.add_argument('--split_train_test',
                    default=False,
                    help="control whether split the dataset")
parser.add_argument('--analyze_labels',
                    default=False,
                    help="analyze the labels (input) so that we can balance the data")
parser.add_argument('--limit_data',
                    default=None,
                    help="Limit the data based on number of paragraph size for debug purposes. None or Int")
# parser.add_argument('--train_splitter_rate',
#                     default=0.6,
#                     help="how much of the data to be used as train")
# parser.add_argument('--eval_question_size_for_recall',
#                     default=2000,
#                     help="how much of the data to be used as train")

# if args.split_train_test is False, data is already splitted,
# file locations of the splitted data: Train Ques/Par Embeddings, Test Ques/Par Embeddings

# TEST/EVAL
parser.add_argument('--test_question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_question_embeddings.hdf5',
                    help="Test/Eval question embeddings data")
parser.add_argument('--test_paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_paragraph_embeddings.hdf5',
                    help="Test/Eval paragraph embeddings data")
parser.add_argument('--test_label_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_q_to_p_mappings.csv',
                    help="Test/Eval paragraph embeddings data")

# TEST/EVAL RECALL
parser.add_argument('--test_recall_question_embeddings',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_recall_question_embeddings.hdf5',
                    help="Test/Eval question embeddings data for recall")

parser.add_argument('--test_recall_paragraph_embeddings',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_paragraph_embeddings.hdf5',
                    help="Test/Eval paragraph embeddings data for recall")

# TRAIN
parser.add_argument('--train_question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_question_embeddings.hdf5',
                    help="Train question embeddings data")
parser.add_argument('--train_paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_paragraph_embeddings.hdf5',
                    help="Train paragraph embeddings data")
parser.add_argument('--train_label_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_q_to_p_mappings.csv',
                    help="Train paragraph embeddings data")

# DATA to be predicted (ALL QUESTIONS)
parser.add_argument('--pretrained_embedding_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
                    help="pretrained embeddings file")


def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    with tf.name_scope('l2_form') as scope:
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True, name='square_sum')
        norm = tf.sqrt(tf.maximum(square_sum, tf.keras.backend.epsilon()), name='norm')
    return norm


def pairwise_cosine_sim(A, B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions
    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point
    """
    with tf.name_scope('pairwise_cosine_sim') as scope:
        A_mag = l2_norm(A, axis=2)
        B_mag = l2_norm(B, axis=2)
        num = tf.keras.backend.batch_dot(A, tf.keras.backend.permute_dimensions(B, (0, 2, 1)))
        den = (A_mag * tf.keras.backend.permute_dimensions(B_mag, (0, 2, 1)))
        dist_mat = num / den
    return dist_mat


def calculate_recalls(questions, paragraphs, labels, params, k=None, extract_type=1):
    """
     question [n x d] tensor of n rows with d dimensions
     paragraphs [m x d] tensor of n rows with d dimensions
     params config
     returns:
     loss : scalar value
     """

    with tf.name_scope('recall_loss') as scope:
        recalls = []  # tf.zeros([len(params.recall_top_k), 1], tf.float32)

        # in order to support batch_size feature, we expanded dims for 1
        paragraphs = tf.expand_dims(paragraphs, axis=0)
        labels = tf.expand_dims(labels, axis=0)
        questions = tf.expand_dims(questions, axis=0)
        number_of_questions = tf.to_int64(tf.shape(questions)[1])
        # now question, paragraphs pairwise calculation
        distances = pairwise_cosine_sim(questions, paragraphs)
        for _k in [k] if k is not None else params.recall_top_k:
            with tf.name_scope('top_k_{}'.format(_k)) as k_scope:
                # find the top_k paragraphs for each question
                top_k = tf.nn.top_k(distances, k=_k, name='top_k_top_k_{}'.format(_k))

                # is groundtruth label is in these top_k paragraph
                equals = tf.equal(top_k.indices, labels, name='equal_top_k_{}'.format(_k))

                # cast the equals to int32 to count the non zero ones because if it is equal,
                # there is only one 1 for each question among paragraphs,
                # then label is in top k
                casted_equal = tf.cast(equals, dtype=tf.int32, name='casted_equal_top_k_{}'.format(_k))
                final_equals_non_zero = tf.squeeze(
                    tf.count_nonzero(casted_equal, axis=2, name='sq_top_k_{}'.format(_k)))

                # get the details of true question - paragraph
                indx_of_questions_that_has_the_correct_paragraphs = tf.reshape(
                    tf.squeeze(tf.where(tf.equal(final_equals_non_zero, extract_type))), shape=[-1, 1])
                top_k_values = tf.reshape(tf.squeeze(top_k.values), shape=[-1, 1])
                cos_values_of_that_has_the_correct_paragraphs = tf.reshape(tf.to_float(tf.gather(top_k_values,
                                                                                                 indx_of_questions_that_has_the_correct_paragraphs)),
                                                                           shape=[-1, 1])
                label_values = tf.reshape(tf.squeeze(labels), shape=[-1, 1])
                label_values_of_that_has_the_correct_paragraphs = tf.reshape(tf.to_float(tf.gather(label_values,
                                                                                                   indx_of_questions_that_has_the_correct_paragraphs)),
                                                                             shape=[-1, 1])
                question_index_labels_and_scores_that_has_the_correct_paragraphs = tf.concat(
                    [tf.reshape(tf.to_float(indx_of_questions_that_has_the_correct_paragraphs),shape=[-1,1]),
                     label_values_of_that_has_the_correct_paragraphs,
                     cos_values_of_that_has_the_correct_paragraphs
                     ], axis=1)



                total_founds_in_k = tf.reduce_sum(final_equals_non_zero)
                recalls.append(total_founds_in_k)

        recalls = tf.stack(recalls)
        best_possible_score = len(params.recall_top_k) * number_of_questions
        current_score = tf.reduce_sum(recalls)
        loss = 1 - (current_score / best_possible_score)

    return loss, recalls, (
                recalls / number_of_questions), number_of_questions, question_index_labels_and_scores_that_has_the_correct_paragraphs


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

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
            # Define the model
            tf.logging.info("Creating the model...")
            config = tf.estimator.RunConfig(tf_random_seed=params.seed,
                                            model_dir=args.model_dir,
                                            save_summary_steps=params.save_summary_steps)
            estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

            if args.is_train:
                # Train the model
                tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
                estimator.train(lambda: train_input_fn(file_paths['train_question_embeddings'],
                                                       file_paths['train_paragraph_embeddings'],
                                                       file_paths['train_paragraph_labels'],
                                                       params))

            if args.is_test:
                # Evaluate the model on the test set
                tf.logging.info("Evaluation on test set.")
                res = estimator.evaluate(lambda: test_input_fn(file_paths['test_question_embeddings'],
                                                               file_paths['test_paragraph_embeddings'],
                                                               file_paths['test_paragraph_labels'],
                                                               params))
                # # res = estimator.evaluate(lambda : test_recall_input_fn(file_paths['test_recall_question_embeddings'],
                # #                                                file_paths['test_recall_paragraph_embeddings'],
                # #                                                params))
                for key in res:
                    print("{}: {}".format(key, res[key]))

            if args.is_prediction_for_evaluation:
                tf.logging.info("Prediction on eval question set.")
                questions = ds.get_question_embeddings(False, file_paths['test_question_embeddings'], params)

                # extract labels from paragraphs because we embedded labels and paragraphs into paragraphs tensor [estimator api]
                labels_ = questions[:, params.embedding_dim:params.embedding_dim + 1]
                labels = tf.cast(labels_, dtype=tf.int32)
                questions = questions[:, :params.embedding_dim]

                predictions = estimator.predict(lambda: live_input_fn(True, questions, params))
                predictions = np.array(list(predictions))

                # Evaluate the model on the test set
                tf.logging.info("Recall evaluation")
                predictions = questions = predictions[:, :params.embedding_dim]
                paragraphs = ds.get_embeddings(file_paths['paragraph_embeddings'])

                extract_type=0
                extract_name='test'

                eval_loss, recalls, normalized_recalls, number_of_questions, q_index_and_cos = \
                    calculate_recalls(questions, paragraphs, labels, params,
                                      1 if args.is_recall_comparision_with_baseline else None, extract_type)
                predictions = np.append(predictions, labels_, axis=1)

                file_name_prefix = 'result_eval'
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    print(10 * '*')
                    print("Question size: {}".format(sess.run(number_of_questions)))
                    print("Eval Recall Loss:{}".format(sess.run(eval_loss)))
                    print("Recalls:{}".format(params.recall_top_k))
                    print("Recall Values:{}".format(sess.run(recalls)))
                    print("Normalized Recall:{}".format(sess.run(normalized_recalls)))
                    if args.is_recall_comparision_with_baseline:
                        qic = sess.run(q_index_and_cos)
                        _e = args.pretrained_embedding_file.rpartition(os.path.sep)
                        path_e = _e[0]
                        if extract_type==1:
                            new_embedding_embed_file = os.path.join(path_e, '{}.csv'.format(extract_name))
                        else:
                            new_embedding_embed_file = os.path.join(path_e, '{}_not.csv'.format(extract_name))
                        np.savetxt(new_embedding_embed_file, qic, delimiter=",", header="q_id,p_id(label),cos")

            if args.is_prediction_for_live:
                tf.logging.info("Prediction on live question set.")
                predictions = estimator.predict(lambda: live_input_fn(False, file_paths['test_question_embeddings'], params))
                predictions = np.array(list(predictions))
                file_name_prefix = 'result_live'

            if args.is_dump_predictions:
                _e = args.pretrained_embedding_file.rpartition(os.path.sep)
                path_e = _e[0]
                new_embedding_embed_file = os.path.join(path_e, file_name_prefix + _e[2].replace('train', ''))
                dump_embeddings(predictions, new_embedding_embed_file)

    print('Done')
