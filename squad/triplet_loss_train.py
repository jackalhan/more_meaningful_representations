"""Train the model"""

import argparse
import os

import tensorflow as tf
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from triplet_loss.input_fn import train_input_fn
from triplet_loss.input_fn import test_input_fn
from triplet_loss.input_fn import live_input_fn
from triplet_loss.model_fn import model_fn
from triplet_loss.utils import Params, train_test_splitter, dump_embeddings, analyze_labes
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function',
                    help="Experiment directory containing params.json")
parser.add_argument('--embeddings_file', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
                    help="embeddings_file")
parser.add_argument('--labels_file', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_q_to_p_mappings.csv',
                    help="labels_file")
parser.add_argument('--split_train_test', default=False,
                    help="control whether split the dataset")
parser.add_argument('--analyze_labels', default=False,
                    help="analyze the labels (input) so that we can balance the data")
parser.add_argument('--is_debug', default=False,
                    help="analyze the labels (input) so that we can balance the data")
parser.add_argument('--train_splitter_rate', default=0.6,
                    help="how much of the data to be used as train")
parser.add_argument('--test_embeddings_file',  default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_question_embeddings.hdf5',
                    help="embeddings_file")
parser.add_argument('--test_labels_file', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_q_to_p_mappings.csv',
                    help="labels_file")
parser.add_argument('--train_embeddings_file',  default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_question_embeddings.hdf5',
                    help="embeddings_file")
parser.add_argument('--train_labels_file', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_q_to_p_mappings.csv',
                    help="labels_file")
parser.add_argument('--live_embeddings_file',  default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_paragraph_embeddings.hdf5',
                    help="embeddings_file")

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

    if args.is_debug:
        file_paths = train_test_splitter(args.embeddings_file, args.labels_file, args.train_splitter_rate, K=analysis['K'])
        params.train_size = file_paths['train_size']
        params.eval_size = file_paths['eval_size']
        params.data_dim = file_paths['data_dim']
        params.num_labels = file_paths['num_labels']
        params.save(json_path)
    else:
        if args.split_train_test:
            file_paths = train_test_splitter(args.embeddings_file, args.labels_file, args.train_splitter_rate, False)
            params.train_size = file_paths['train_size']
            params.eval_size = file_paths['eval_size']
            params.data_dim = file_paths['data_dim']
            params.num_labels = file_paths['num_labels']
            params.save(json_path)
        else:
            file_paths = {}
            file_paths['train_embeddings'] = args.train_embeddings_file
            file_paths['train_labels'] = args.train_labels_file

            file_paths['test_embeddings'] = args.test_embeddings_file
            file_paths['test_labels'] = args.test_labels_file


    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(file_paths['train_embeddings'], file_paths['train_labels'], params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(file_paths['test_embeddings'], file_paths['test_labels'], params))
    for key in res:
        print("{}: {}".format(key, res[key]))

    tf.logging.info("Prediction on live question set.")
    predictions = estimator.predict(lambda: live_input_fn(args.embeddings_file, params))
    predictions = np.array(list(predictions))

    _e = args.embeddings_file.rpartition(os.path.sep)
    path_e = _e[0]
    new_embedding_embed_file = os.path.join(path_e, 'result' + _e[2].replace('train', ''))
    dump_embeddings(predictions,new_embedding_embed_file)

    tf.logging.info("Prediction on live paragraph set.")
    predictions = estimator.predict(lambda: live_input_fn(args.live_embeddings_file, params))
    predictions = np.array(list(predictions))
    _e = args.live_embeddings_file.rpartition(os.path.sep)
    path_e = _e[0]
    new_embedding_embed_file = os.path.join(path_e, 'result' + _e[2].replace('train', ''))
    dump_embeddings(predictions, new_embedding_embed_file)
    print('Done')