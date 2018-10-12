"""Train the model"""

import os
import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.parser as parser
import numpy as np
from helper.utils import load_from_pickle, vocabulary_processor, fit_vocab_to_documents,  tokenize_contexts, load_word_embeddings, prepare_squad_objects, Params, define_pre_executions,dump_embeddings, save_as_pickle, save_as_shelve, load_embeddings
from helper.estimator_input_fn import DataBuilder
from helper.estimator_model_fn import model_fn
import math
from tensorflow.python.keras.preprocessing import sequence
from itertools import tee

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

def execute_non_conv_pipeline(params, base_data_path, config, tf, databuilder):
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    if not params.executor["is_prediction"]:
        _train_input_fn = lambda: databuilder.train_input_fn()
        if params.executor["recall_calculation_for"] == 'test':
            _recall_input_fn = lambda: databuilder.test_recall_input_fn()
        else:
            _recall_input_fn = lambda: databuilder.train_recall_input_fn()


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
            #estimator.train(_train_input_fn, max_steps=1)

                # -------------------------------------
                # Train and Test : Train
                # -------------------------------------
            # train_spec = tf.estimator.TrainSpec(_train_input_fn,
            #                                     max_steps= params.model['num_epochs'] * math.ceil(params.files['splitter']['train_size'] / params.model['batch_size']))

            train_spec = tf.estimator.TrainSpec(_train_input_fn)


            # -------------------------------------
            # Test the model
            # -------------------------------------
            tf.logging.info("Evaluation loss and recall loss on test set.")
            tf.logging.info(
                "Evaluation loss on test set with a size of {} ".format(params.files["splitter"]["test_size"]))
            tf.logging.info(
                "Evaluation recall loss on test set with a size of {} ".format(params.files["splitter"]["test_subset_size"]))

            # -------------------------------------
            # Baseline Eval for Initial model
            # -------------------------------------
                # -------------------------------------
                # Then Test
                # -------------------------------------
            #estimator.evaluate(_recall_input_fn)

                # -------------------------------------
                # Train and Test : Test
                # -------------------------------------
            test_spec = tf.estimator.EvalSpec(_recall_input_fn) #steps=params.model['num_epochs']

            # -------------------------------------
            # Train and Test
            # -------------------------------------
            tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)


            if params.executor["is_prediction_during_training"]:

                predictions = estimator.predict(lambda: databuilder.predict_input_fn())
                predictions = np.array(list(predictions))
                dump_embeddings(predictions, os.path.join(base_data_path, "improved_" + params.files["prediction"][
                    "source_embeddings"]))
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
                result_as_dict_list = estimator.predict(_recall_input_fn)
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
        predictions = estimator.predict(lambda: databuilder.predict_input_fn())
        predictions = np.array(list(predictions))
        dump_embeddings(predictions,
                        os.path.join(base_data_path, "improved_" + params.files["prediction"]["source_embeddings"]))


def execute_conv_pipeline(params, base_data_path, config, tf, databuilder):

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    if not params.executor["is_prediction"]:
        _train_input_fn = lambda: databuilder.train_input_fn()
        if params.executor["recall_calculation_for"] == 'test':
            _recall_input_fn = lambda: databuilder.test_recall_input_fn()
        else:
            _recall_input_fn = lambda: databuilder.train_recall_input_fn()
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
            #estimator.train(_train_input_fn, max_steps=1)

            # -------------------------------------
            # Train and Test : Train
            # -------------------------------------
            train_spec = tf.estimator.TrainSpec(_train_input_fn,
                                                max_steps=params.model['num_epochs'] * math.ceil(
                                                    params.files['splitter']['train_size'] / params.model['batch_size']))

            # -------------------------------------
            # Test the model
            # -------------------------------------
            tf.logging.info("Evaluation loss and recall loss on test set.")
            tf.logging.info(
                "Evaluation Floss on test set with a size of {} ".format(params.files["splitter"]["test_size"]))
            tf.logging.info(
                "Evaluation recall loss on test set with a size of {} ".format(
                    params.files["splitter"]["test_subset_size"]))

            # -------------------------------------
            # Baseline Eval for Initial model
            # -------------------------------------
            # -------------------------------------
            # Then Test
            # -------------------------------------
            #estimator.evaluate(_recall_input_fn)

            # -------------------------------------
            # Train and Test : Test
            # -------------------------------------
            test_spec = tf.estimator.EvalSpec(_recall_input_fn)  # steps=params.model['num_epochs']

            # -------------------------------------
            # Train and Test
            # -------------------------------------
            tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)


            if params.executor["is_prediction_during_training"]:
                predictions = estimator.predict(lambda: databuilder.predict_input_fn())
                predictions = np.array(list(predictions))
                dump_embeddings(predictions, os.path.join(base_data_path, "improved_" + params.files["prediction"]["source_embeddings"]))
        else:
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
                result_as_dict_list = estimator.predict(_recall_input_fn)
                tf.logging.info("-------> Epoch: {} Predict is completed".format(ep))
                data_dict = prepare_dict_to_print(result_as_dict_list, params, data_dict, ep)
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
        predictions = estimator.predict(lambda: databuilder.predict_input_fn())
        predictions = np.array(list(predictions))
        dump_embeddings(predictions,
                        os.path.join(base_data_path, "improved_" + params.files["prediction"]["source_embeddings"]))

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
    db = DataBuilder(base_data_path, params, ['train', params.executor["recall_calculation_for"] + '_recall', 'predict'],True)
    params = db.params
    if params.model['model_type'].lower() == 'conv':
        execute_conv_pipeline(params, base_data_path, config,tf, db)
    else:
        execute_non_conv_pipeline(params, base_data_path, config, tf, db)


    print('Done')