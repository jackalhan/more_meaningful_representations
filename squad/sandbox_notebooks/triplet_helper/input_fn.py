"""Create the input data pipeline using `tf.data`"""

import triplet_loss.my_dataset as ds
import tensorflow as tf

def train_input_fn(question_embeddings_file, paragraph_embeddings_file, labels_file, params):
    """Train input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = ds.get_dataset(question_embeddings_file, paragraph_embeddings_file, labels_file, [params.embedding_dim+1,], [params.embedding_dim,])
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(question_embeddings_file, paragraph_embeddings_file,labels_file, params):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    dataset = ds.get_dataset(question_embeddings_file, paragraph_embeddings_file, labels_file, [params.embedding_dim+1,], [params.embedding_dim,])
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def test_recall_input_fn(question_embeddings_file, paragraph_embeddings_file, params):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    questions, paragraphs = ds.get_dataset_no_batch(question_embeddings_file, paragraph_embeddings_file)
    # dataset = dataset.batch(params.batch_size)
    # dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    questions = questions.batch(params.eval_question_size_for_recall)
    #questions = questions.prefetch(1)
    paragraphs = paragraphs.batch(params.num_labels)
    #paragraphs = paragraphs.prefetch(1)
    return questions, paragraphs

def live_input_fn(is_cached, question_embeddings, params):
    """Test input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    if is_cached:
        dataset = ds.get_dataset_from_cache(question_embeddings)
    else:
        dataset = ds.get_dataset(question_embeddings, None, None, [params.embedding_dim+1,], None, False)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
