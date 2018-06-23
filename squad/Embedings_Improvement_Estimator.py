"""Train the model"""

import os
import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.parser as parser
import numpy as np
from helper.utils import Params, define_pre_executions,dump_embeddings,create_execution_name
from helper.estimator_input_fn import train_input_fn, test_input_fn, live_input_fn
from helper.estimator_model_fn import model_fn
import math

if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.get_parser().parse_args()
    assert os.path.isfile(args.json_path), "No json configuration file found at {}".format(args.json_path)
    params = Params(args.json_path)

    # split the files or use the provided files
    params = define_pre_executions(params, args.json_path)
    base_data_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])
    model_save_path = os.path.join(params.executor['model_dir'], params.executor['save_dir'], create_execution_name(params))

    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=params.model["seed"],
                                    model_dir=model_save_path,
                                    save_summary_steps= params.model['num_epochs'] *math.ceil(params.files['splitter']['train_size'] / params.model['batch_size']))

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    _train_input_fn = lambda: train_input_fn(base_data_path,
                                             params)
    _test_input_fn = lambda: test_input_fn(base_data_path,
                                           params)

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
    #
    train_spec = tf.estimator.TrainSpec(_train_input_fn, max_steps= params.model['num_epochs'] * math.ceil(params.files['splitter']['train_size'] / params.model['batch_size']))
    #

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

    # -------------------------------------
    # Predict the model
    # -------------------------------------
    if params.executor["is_prediction"]:
        tf.logging.info("Results (Improved Embeddings ? ) from the trained model")
        predictions = estimator.predict(lambda: live_input_fn(base_data_path, params))
        predictions = np.array(list(predictions))
        dump_embeddings(predictions, "improved_" + params.files["pre_trained_files"]["question_embeddings"])

    print('Done')
