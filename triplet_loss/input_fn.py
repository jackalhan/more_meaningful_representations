"""Create the input data pipeline using `tf.data`"""

import triplet_loss.my_dataset as ds


def train_input_fn(embeddings_file, labels_file, params):
    """Train input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = ds.get_dataset(embeddings_file, labels_file, [params.data_dim,])
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(embeddings_file, labels_file, params):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = ds.get_dataset(embeddings_file, labels_file, [params.data_dim,])
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def live_input_fn(embeddings_file, params):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = ds.get_dataset(embeddings_file, None, [params.data_dim,], False)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
