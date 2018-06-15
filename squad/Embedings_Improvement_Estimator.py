"""Train the model"""

import os

import tensorflow as tf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.parser as parser
import numpy as np
from helper.utils import load_embeddings, Params, define_pre_executions,get_question_and_paragraph_embeddings,calculate_recalls,dump_embeddings
from helper.estimator_input_fn import train_input_fn, test_input_fn, live_input_fn
from helper.estimator_model_fn import model_fn

if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.get_parser().parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # split the files or use the provided files
    file_paths = define_pre_executions(args, params, json_path)
    tf.logging.info("Creating the model...")
    tf.set_random_seed(params.seed)

    config = tf.estimator.RunConfig(tf_random_seed=params.seed,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    if args.is_train:
        # Train the model
        tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
        estimator.train(lambda: train_input_fn(file_paths['train_question_embeddings'],
                                               file_paths['train_paragraph_embeddings'],
                                               params))

    if args.is_test:
        # Evaluate the model's loss on the test set
        tf.logging.info("Evaluation loss on test set.")
        res = estimator.evaluate(lambda: test_input_fn(file_paths['test_question_embeddings'],
                                                       file_paths['test_paragraph_embeddings'],
                                                       params))
        for key in res:
            print("{}: {}".format(key, res[key]))

    if args.is_test_for_recall:
        tf.logging.info("Evaluation recall on test set with a size of {} ".format(params.eval_question_size_for_recall))

        # Evaluate the model's recall on the test set
        # 5k questions, ~20k paragraphs, can not be handled in estimator api (number of questions and number of paragraphs are not same)
        # In order to accelarete the calculation, I prefer doing the following
        predictions = estimator.predict(lambda: live_input_fn(False, file_paths['test_question_embeddings'], params))
        predictions = np.array(list(predictions))

        input_questions = tf.placeholder(tf.float32, [None, params.embedding_dim], name='input_questions')
        input_labels = tf.placeholder(tf.int32, [None, 1], name='input_labels')
        input_recall_paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim], name='input_recall_paragraphs')
        paragraphs = tf.nn.l2_normalize(input_recall_paragraphs, axis=1)
        avg_recall, recalls, normalized_recalls, number_of_questions, q_index_and_cos = calculate_recalls(
            input_questions,
            input_recall_paragraphs,
            input_labels,
            params)

        testing_question_embeddings, testing_paragraph_embeddings = get_question_and_paragraph_embeddings(True, False,
                                                                                                          predictions,
                                                                                                          file_paths[
                                                                                                              'test_paragraph_embeddings'],
                                                                                                          params)
        testing_labels = testing_question_embeddings[:, params.embedding_dim:params.embedding_dim + 1]
        testing_question_embeddings = testing_question_embeddings[:, :params.embedding_dim]
        all_paragraph_embeddings_for_recall = load_embeddings(args.paragraph_embeddings_file)

        # Evaluate the model on the test set
        tf.logging.info("Recall evaluation")


        file_name_prefix = 'result_eval'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _avg_recall_, _recalls_, _normalized_recalls_, _number_of_questions_ = sess.run([avg_recall, recalls, normalized_recalls, number_of_questions],
                                                                                            feed_dict={input_questions:testing_question_embeddings,
                                                                                                                 input_labels:testing_labels,
                                                                                                                 input_recall_paragraphs:all_paragraph_embeddings_for_recall})
            print(10 * '*')
            print("Question size: {}".format(_number_of_questions_))
            print("Avg Recall Loss:{}".format(_avg_recall_))
            print("Recalls:{}".format(params.recall_top_k))
            print("Recall Values:{}".format(_recalls_))
            print("Normalized Recall:{}".format(_normalized_recalls_))

    if args.is_prediction_for_live:
        tf.logging.info("Prediction on live question set.")
        predictions = estimator.predict(lambda: live_input_fn(False, file_paths['test_question_embeddings'], params))
        predictions = np.array(list(predictions))


    if args.is_dump_predictions:
        _e = args.pretrained_embedding_file.rpartition(os.path.sep)
        file_name_prefix = 'result_live'
        path_e = _e[0]
        new_embedding_embed_file = os.path.join(path_e, file_name_prefix + _e[2].replace('train', ''))
        dump_embeddings(testing_question_embeddings, new_embedding_embed_file)

    print('Done')
