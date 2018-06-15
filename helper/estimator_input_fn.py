"""Create the input data pipeline using `tf.data`"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.estimator_dataset as ds

def train_input_fn(question_embeddings_file, paragraph_embeddings_file, params):
    """Train input function for the dataset.

    Args:
        question_embeddings_file: (string) path to the question_embeddings_file
        paragraph_embeddings_file:  (string) path to the paragraph_embeddings_file
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = ds.get_dataset(question_embeddings_file,
                             paragraph_embeddings_file,
                             params.embedding_dim,
                             including_target=True)
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(question_embeddings_file, paragraph_embeddings_file, params):
    """Test input function for the dataset.

    Args:
        question_embeddings_file: (string) path to the question_embeddings_file
        paragraph_embeddings_file:  (string) path to the paragraph_embeddings_file
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    dataset = ds.get_dataset(question_embeddings_file,
                             paragraph_embeddings_file,
                             params.embedding_dim,
                             including_target=True)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def live_input_fn(is_cached, question_embeddings, params):
    """Test input function for the dataset.

    Args:
        question_embeddings_file: (string) path to the question_embeddings_file
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    if is_cached:
        dataset = ds.get_dataset_from_cache(question_embeddings)
    else:
        dataset = ds.get_dataset(question_embeddings,
                                 None,
                                 params.embedding_dim,
                                 including_target=False)

    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
