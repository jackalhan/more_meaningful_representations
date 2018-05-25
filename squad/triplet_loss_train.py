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
parser.add_argument('--train_splitter_rate',
                    default=0.6,
                    help="how much of the data to be used as train")

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
                                             args.train_splitter_rate,
                                             args.limit_data)
            params.train_size = file_paths['train_question_size']
            params.eval_size = file_paths['eval_question_size']
            params.data_dim = file_paths['data_dim']
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

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(file_paths['train_question_embeddings'],
                                           file_paths['train_paragraph_embeddings'],
                                           file_paths['train_paragraph_labels'],
                                           params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(file_paths['test_question_embeddings'],
                                                   file_paths['test_paragraph_embeddings'],
                                                   file_paths['test_paragraph_labels'],
                                                   params))
    for key in res:
        print("{}: {}".format(key, res[key]))

    tf.logging.info("Prediction on live question set.")
    predictions = estimator.predict(lambda: live_input_fn(args.pretrained_embedding_file, params))
    predictions = np.array(list(predictions))

    _e = args.pretrained_embedding_file.rpartition(os.path.sep)
    path_e = _e[0]
    new_embedding_embed_file = os.path.join(path_e, 'result' + _e[2].replace('train', ''))
    dump_embeddings(predictions,new_embedding_embed_file)

    print('Done')