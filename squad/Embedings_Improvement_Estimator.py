"""Train the model"""

import os
import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.parser as parser
import numpy as np
from helper.utils import load_from_pickle, vocabulary_processor, fit_vocab_to_documents,  tokenize_contexts, load_word_embeddings, prepare_squad_objects, Params, define_pre_executions,dump_embeddings, save_as_pickle, save_as_shelve, load_embeddings
from helper.estimator_input_fn import train_input_fn, test_input_fn, live_input_fn
from helper.estimator_model_fn import model_fn
import math
from tensorflow.python.keras.preprocessing import sequence
from itertools import tee

def predictor(estimator, base_data_path, params):
    predictions = estimator.predict(lambda: live_input_fn(base_data_path, params))
    predictions = np.array(list(predictions))
    return predictions

def prepare_dict_to_print(result_as_dict_list, params, data_dict, epoch):
    if data_dict is None or not data_dict :
        data_dict = {}
    sub_data_dict = {}

    result_as_dict_list = list(result_as_dict_list)
    avg_recall_after_model = np.array([o['avg_recall_after_model'] for o in result_as_dict_list])
    sub_data_dict['avg_recall_after_model'] =  avg_recall_after_model[0]
    normalized_recalls_after_model = np.array([o['normalized_recalls_after_model'] for o in result_as_dict_list])
    sub_data_dict['normalized_recalls_after_model'] = normalized_recalls_after_model[0]
    sub_data_dict["distance_from_after_model_q_to_p"] = np.array([o['distance_from_after_model_q_to_p'] for o in result_as_dict_list])

    avg_recall_before_model = np.array([o['avg_recall_before_model'] for o in result_as_dict_list])
    data_dict['avg_recall_before_model'] = avg_recall_before_model[0]
    normalized_recalls_before_model = np.array([o['normalized_recalls_before_model'] for o in result_as_dict_list])
    data_dict['normalized_recalls_before_model'] = normalized_recalls_before_model[0]
    data_dict["distance_from_before_model_q_to_p"]= np.array([o['distance_from_before_model_q_to_p'] for o in result_as_dict_list])

    sub_data_dict["delta_before_after_model"] = np.array(
        [o['delta_before_after_model'] for o in result_as_dict_list])


    sub_data_dict['embeddings'] = np.array([o['embeddings'] for o in result_as_dict_list])
    data_dict['actual_labels'] = np.array([o['actual_labels'] for o in result_as_dict_list])

    are_founds_before = np.array([o['are_founds_before'] for o in result_as_dict_list])
    closest_labels_before = np.array([o['closest_labels_before'] for o in result_as_dict_list])
    distances_before = np.array([o['distances_before'] for o in result_as_dict_list])

    are_founds_after = np.array([o['are_founds_after'] for o in result_as_dict_list])
    closest_labels_after = np.array([o['closest_labels_after'] for o in result_as_dict_list])
    distances_after = np.array([o['distances_after'] for o in result_as_dict_list])

    sub_data_each_k_dict_after = {}
    sub_data_each_k_dict_before = {}
    end = 0
    for k in range(1, params.executor["debug_top_k"]+1):
        start = end
        end = start + k

        sub_data_each_k_dict_before[str(k)] = {}
        sub_data_each_k_dict_before[str(k)]['are_founds_before'] = are_founds_before[:, k - 1:k]
        sub_data_each_k_dict_before[str(k)]['closest_labels_before'] = closest_labels_before[:, start:end]
        sub_data_each_k_dict_before[str(k)]['distances_before'] = distances_before[:, start:end]

        sub_data_each_k_dict_after[str(k)]= {}
        sub_data_each_k_dict_after[str(k)]['are_founds_after'] = are_founds_after[:,k-1:k]
        sub_data_each_k_dict_after[str(k)]['closest_labels_after'] = closest_labels_after[:, start:end]
        sub_data_each_k_dict_after[str(k)]['distances_after'] = distances_after[:, start:end]
    sub_data_dict['top_k'] = sub_data_each_k_dict_after
    data_dict['top_k'] = sub_data_each_k_dict_before
    data_dict[str(epoch)] = sub_data_dict
    return data_dict

def execute_non_conv_pipeline(params, base_data_path, config, tf):
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    if not params.executor["is_prediction"]:

        _train_input_fn = lambda: train_input_fn(base_data_path,
                                                 params)
        _test_input_fn = lambda: test_input_fn(base_data_path,
                                               params)

        if not params.executor["is_debug_mode"]:
            # -------------------------------------
            # Train the model
            # -------------------------------------
            tf.logging.info("Starting training for {} epoch(s).".format(params.model["num_epochs"]))
            tf.logging.info(
                "Train loss on train set with a size of {} ".format(params.files["splitter"]["train_size"]))
                # -------------------------------------
                # First Train
                # -------------------------------------
            estimator.train(_train_input_fn, max_steps=1)

                # -------------------------------------
                # Train and Test : Train
                # -------------------------------------
            train_spec = tf.estimator.TrainSpec(_train_input_fn,
                                                max_steps= params.model['num_epochs'] * math.ceil(params.files['splitter']['train_size'] / params.model['batch_size']))


            # -------------------------------------
            # Test the model
            # -------------------------------------
            tf.logging.info("Evaluation loss and recall loss on test set.")
            tf.logging.info(
                "Evaluation Floss on test set with a size of {} ".format(params.files["splitter"]["test_size"]))
            tf.logging.info(
                "Evaluation recall loss on test set with a size of {} ".format(params.files["splitter"]["test_subset_size"]))

            # -------------------------------------
            # Baseline Eval for Initial model
            # -------------------------------------
                # -------------------------------------
                # Then Test
                # -------------------------------------
            estimator.evaluate(_test_input_fn)

                # -------------------------------------
                # Train and Test : Test
                # -------------------------------------
            test_spec = tf.estimator.EvalSpec(_test_input_fn) #steps=params.model['num_epochs']

            # -------------------------------------
            # Train and Test
            # -------------------------------------
            tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)

            # for key in res:
            #     print("{}: {}".format(key, res[key]))

            # ----------------------------------------
            # Save Model
            # ---------------------------------------
            # Added this for solving the issue
            # def serving_input_receiver_fn():
            #
            #     feature_spec = {'image/encoded': tf.FixedLenFeature(shape=[],
            #                                                         dtype=tf.string)}
            #
            #     serialized_tf_example = tf.placeholder(
            #         dtype=tf.string)
            #
            #     receiver_tensors = {'examples': serialized_tf_example}
            #
            #     features = tf.parse_example(serialized_tf_example, feature_spec)
            #     jpegs = features['image/encoded']
            #     fn = lambda x : ds.get_dataset(os.path.join(base_data_path, params.files['pre_trained_files']['question_embeddings']),
            #                      None,
            #                      params.files['pre_trained_files']['embedding_dim'],
            #                      including_target=False)
            #
            #     images = tf.map_fn(fn, jpegs, dtype=tf.float32)
            #
            #     return tf.estimator.export.ServingInputReceiver(images, receiver_tensors)
            #
            # estimator.export_savedmodel(model_save_path, serving_input_receiver_fn)

            if params.executor["is_prediction_during_training"]:
                predictions = predictor(estimator, base_data_path, params)
                dump_embeddings(predictions, "improved_" + params.files["prediction"]["question_embeddings"])
        else:
            tf.logging.info(10 * '*')
            tf.logging.info("Starting debugging for {} epoch(s).".format(params.model["num_epochs"]))
            data_dict = {}
            data_dict['epochs'] = params.model["num_epochs"]
            for ep in range(1,params.model["num_epochs"] + 1):
                tf.logging.info("-------> Epoch: {}".format(ep))
                # if ep < 1:
                #     estimator.train(_train_input_fn, max_steps=1)
                # else:
                estimator.train(_train_input_fn)
                tf.logging.info("-------> Epoch: {} Train is completed".format(ep))
                result_as_dict_list = estimator.predict(lambda: test_input_fn(base_data_path, params))
                tf.logging.info("-------> Epoch: {} Predict is completed".format(ep))
                data_dict = prepare_dict_to_print(result_as_dict_list, params, data_dict,ep)
                tf.logging.info("-------> Epoch: {} data_dict is completed".format(ep))

            save_as_pickle(data_dict, os.path.join(model_save_path, 'debug_dict.pkl'))
            save_as_shelve(data_dict, os.path.join(model_save_path, 'debug_dict.slv'))
            tf.logging.info('Dict objs are saved.')
            tf.logging.info(10 * '*')


    # -------------------------------------
    # Prediction
    # -------------------------------------
    else:
        # ALL PARAMETERS SHOULD BE SET SAME WITH THE SAVED MODEL :(
        # I AM GOING TO HANDLE HOW TO SAVE IT WITH THE PARAM CONFS.
        predictions = predictor(estimator, base_data_path, params)
        dump_embeddings(predictions, "improved_" + params.files["prediction"]["question_embeddings"])


def execute_conv_pipeline(params, base_data_path, config, tf):

    """
    START: DATA PREPARATION
    """

    train_question_label_indx = load_embeddings(os.path.join(base_data_path,
                                                                  params.files['train_loss']['question_labels']))
    train_question_label_indx = train_question_label_indx.astype(int)

    train_org_questions = load_embeddings(os.path.join(base_data_path,
                                                            params.files['train_loss']['question_embeddings']))
    train_question_labels = load_embeddings(os.path.join(base_data_path,
                                                            params.files['train_loss']['paragraph_embeddings']))


    # TEST RECALL = VALID QUESTIONS ARE GETTING LOADED

    valid_question_label_indx = load_embeddings(os.path.join(base_data_path,
                                                                  params.files['test_subset_recall']['question_labels']))
    valid_question_label_indx = valid_question_label_indx.astype(int)

    valid_org_questions = load_embeddings(os.path.join(base_data_path,
                                                            params.files['test_subset_recall']['question_embeddings']))

    valid_question_labels = load_embeddings(os.path.join(base_data_path,
                                                            params.files['test_subset_recall']['paragraph_embeddings']))



    x_train = load_from_pickle(os.path.join(base_data_path,
                                                            params.files['train_loss']['question_x_train']))

    y_train_paragraph = train_question_labels
    y_train_labels = train_question_label_indx
    del train_question_labels, train_question_label_indx
    print("x_train shape is {}".format(x_train.shape))


    x_valid = load_from_pickle(os.path.join(base_data_path,
                                                            params.files['test_subset_recall']['question_x_valid']))
    y_valid_paragraph = valid_question_labels
    y_valid_labels = valid_question_label_indx
    del valid_question_labels,valid_question_label_indx
    print("x_valid shape is {}".format(x_valid.shape))

    voc_to_indx = load_from_pickle(os.path.join(base_data_path,
                                                            params.files['voc_to_indx']))
    vocab_size = len(voc_to_indx)
    print('Total words: %d' % vocab_size)
    params.files['questions_vocab_size'] = vocab_size

    max_document_len = params.files['max_document_len']


    if params.files['word_embeddings'] is None:
        params.model['conv_embedding_initializer'] = tf.truncated_normal_initializer(seed=params.model['initializer_seed'],
                                                                    stddev=0.1)
    else:
        word_embeddings = load_word_embeddings(os.path.join(base_data_path, params.files['word_embeddings']),
                                               voc_to_indx,
                                               params.files['pre_trained_files']['embedding_dim'])

        def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
            assert dtype is tf.float32
            return word_embeddings
        params.model['conv_embedding_initializer']= my_initializer
    """
    END: DATA PREPARATION
    """
    # ----------------------------------------------------
    """
    START: BUILDING ESTIMATORS
    """
    x_len_train = np.array([min(len(x), max_document_len) for x in x_train])
    x_len_valid = np.array([min(len(x), max_document_len) for x in x_valid])

    def parser(x, length, org, y_paragraph, y_labels):
        features = {"x": x, "len": length, "org": org}
        labels = {"paragraph": y_paragraph, "labels": y_labels}
        return features, labels

    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, train_org_questions, y_train_paragraph, y_train_labels))
        if params.model["shuffle"]:
            dataset = dataset.shuffle(buffer_size=x_train.shape[0])
        dataset = dataset.batch(params.model["batch_size"])
        dataset = dataset.map(parser)
        #dataset = dataset.repeat()
        #dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def test_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_valid, x_len_valid, valid_org_questions, y_valid_paragraph, y_valid_labels))
        dataset = dataset.batch(params.files["splitter"]["test_subset_size"])
        dataset = dataset.map(parser)
        #dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    """
    END: BUILDING ESTIMATORS
    """

    _train_input_fn = lambda: train_input_fn()
    _test_input_fn = lambda: test_input_fn()
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    if params.executor["is_debug_mode"]:
        tf.logging.info(10 * '*')
        tf.logging.info("Starting debugging for {} epoch(s).".format(params.model["num_epochs"]))
        data_dict = {}
        data_dict['epochs'] = params.model["num_epochs"]
        for ep in range(1, params.model["num_epochs"] + 1):
            tf.logging.info("-------> Epoch: {}".format(ep))
            # if ep < 1:
            #     estimator.train(_train_input_fn, max_steps=1)
            # else:
            estimator.train(_train_input_fn)
            tf.logging.info("-------> Epoch: {} Train is completed".format(ep))
            result_as_dict_list = estimator.predict(_test_input_fn)
            tf.logging.info("-------> Epoch: {} Predict is completed".format(ep))
            data_dict = prepare_dict_to_print(result_as_dict_list, params, data_dict, ep)
            tf.logging.info("-------> Epoch: {} data_dict is completed".format(ep))

        save_as_pickle(data_dict, os.path.join(model_save_path, 'debug_dict.pkl'))
        save_as_shelve(data_dict, os.path.join(model_save_path, 'debug_dict.slv'))
        tf.logging.info('Dict objs are saved.')
        tf.logging.info(10 * '*')


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.get_parser().parse_args()
    assert os.path.isfile(args.json_path), "No json configuration file found at {}".format(args.json_path)
    params = Params(args.json_path)

    # split the files or use the provided files
    base_data_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])


    params, model_save_path = define_pre_executions(params, args.json_path, base_data_path)

    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=params.model["seed"],
                                    model_dir=model_save_path,
                                    save_summary_steps= params.model['num_epochs'] *math.ceil(params.files['splitter']['train_size'] / params.model['batch_size']))


    # IF MODEL TYPE IS CONV, START THE PIPELINE ACCORDINGLY
    if params.model['model_type'].lower() == 'conv':
        execute_conv_pipeline(params, base_data_path, config,tf)
    else:
        execute_non_conv_pipeline(params, base_data_path, config, tf)


    print('Done')