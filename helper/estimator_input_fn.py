"""Create the input data pipeline using `tf.data`"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.estimator_dataset as ds
import helper.utils as UTIL

def train_input_fn(base_data_path, params):
    """Train input function for the dataset.

    Args:
        base_data_path: (string) base path for all data
        params: (Params) contains all the details of the execution including names of the data
    """
    dataset = ds.get_dataset(os.path.join(base_data_path, params.files['train_loss']['question_embeddings']),
                             os.path.join(base_data_path, params.files['train_loss']['question_labels']),
                             os.path.join(base_data_path, params.files['train_loss']['paragraph_embeddings']),
                             params.files['pre_trained_files']['embedding_dim'],
                             including_target=True)
    if params.model["shuffle"]:
        dataset = dataset.shuffle(params.files['splitter']["train_size"])  # whole dataset into the buffer
    #dataset = dataset.repeat(1)  # repeat for multiple epochs
    dataset = dataset.batch(params.model["batch_size"])
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(base_data_path, params):
    """Test input function for the dataset.

    Args:
        base_data_path: (string) base path for all data
        params: (Params) contains all the details of the execution including names of the data
    """

    dataset = ds.get_dataset(os.path.join(base_data_path, params.files['test_subset_loss']['question_embeddings']),
                             None,
                             os.path.join(base_data_path, params.files['test_subset_loss']['paragraph_embeddings']),
                             params.files['pre_trained_files']['embedding_dim'],
                             including_target=True)
    dataset = dataset.batch(params.files["splitter"]["test_subset_size"])
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def live_input_fn(base_data_path, params):
    """Live input function for the dataset.

    Args:
        base_data_path: (string) base path for all data
        params: (Params) contains all the details of the execution including names of the data
    """
    dataset = ds.get_dataset(os.path.join(base_data_path, params.files['prediction']['question_embeddings']),
                             None,
                             None,
                             params.files['pre_trained_files']['embedding_dim'],
                             including_target=False)

    dataset = dataset.batch(params.model["batch_size"])
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

