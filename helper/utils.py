"""General utility functions"""

import json
import logging
import pandas as pd
import h5py
import random
import numpy as np
import math
import os
from random import shuffle
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import sys
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def define_pre_executions(params, json_path):
    if params.executor['analyze_labels']:
        analysis = analyze_labels(params.files['pre_trained_files']['question_labels'])
        print(analysis)

    else:
        if params.executor["split_train_set"]:
            params = train_test_splitter(params)

            params.save(json_path)
            print('Done with splitting')

    return params

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def train_test_splitter(params):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
    """
    base_path = os.path.join(params.executor['model_dir'], params.executor['data_dir'])
    pre_trained_question_embeddings_file = os.path.join(base_path, params.files['pre_trained_files']['question_embeddings'])
    pre_trained_paragraph_embeddings_file = os.path.join(base_path, params.files['pre_trained_files']['paragraph_embeddings'])
    pre_trained_question_labels_file = os.path.join(base_path, params.files['pre_trained_files']['question_labels'])

    pre_trained_question_labels = pd.read_csv(pre_trained_question_labels_file)
    pre_trained_question_labels = pre_trained_question_labels['p'].tolist()
    question_embeddings = load_embeddings(pre_trained_question_embeddings_file)
    paragraph_embeddings = load_embeddings(pre_trained_paragraph_embeddings_file)
    num_labels = len(set(pre_trained_question_labels))

    if params.executor['limit_paragraph_size'] is not None:
        # just grab the questions within the size of the paragraph that are limited with this parameter
        num_labels = params.executor['limit_paragraph_size']

    labels_as_list = list(range(num_labels))
    shuffle(labels_as_list)
    #in order to have all the labels in the training set, we need to split them accordingly:
    train_labels, test_labels= list(), list()
    train_ques_embeddings, train_par_embeddings = list(), list()
    test_ques_embeddings, test_par_embeddings = list(), list()
    for i in labels_as_list:
        locations = [_ for _, x in enumerate(pre_trained_question_labels) if x == i]
        shuffle(locations)
        occur = len(locations)
        print(10 * '*')
        print('For p: {}, we have -> {} qs ---> {}'.format(i, occur, locations))
        for_local_train_size = math.ceil(occur * params.files['splitter']['train_split_rate'])
        for_local_train_locations = locations[0:for_local_train_size]
        for_local_train_labels = list()
        for_local_train_ques_embeddings = list()
        for_local_train_par_embeddings = list()

        for _l in for_local_train_locations:
            for_local_train_labels.append(i)
            for_local_train_ques_embeddings.append(question_embeddings[_l])
            for_local_train_par_embeddings.append(paragraph_embeddings[i])
        train_labels.extend(for_local_train_labels)
        train_ques_embeddings.extend(for_local_train_ques_embeddings)
        train_par_embeddings.extend(for_local_train_par_embeddings)
        print('Train Size {} ---> {}'.format(for_local_train_size, for_local_train_locations))


        for_local_test_locations = locations[for_local_train_size:]
        for_local_test_size = len(for_local_test_locations)
        for_local_test_labels = list()
        for_local_test_ques_embeddings = list()
        for_local_test_par_embeddings = list()
        for _l in for_local_test_locations:
            for_local_test_labels.append(i)
            for_local_test_ques_embeddings.append(question_embeddings[_l])
            for_local_test_par_embeddings.append(paragraph_embeddings[i])

        test_labels.extend(for_local_test_labels)
        test_ques_embeddings.extend(for_local_test_ques_embeddings)
        test_par_embeddings.extend(for_local_test_par_embeddings)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    #assert num_labels == len(set(test_labels)), "Actual Num of Labels: {} vs Test Num of Labels {}".format(num_labels, len(set(test_labels)))

    train = list(zip(train_ques_embeddings, train_labels, train_par_embeddings))
    test = list(zip(test_ques_embeddings, test_labels, test_par_embeddings))

    random.shuffle(train)
    random.shuffle(test)
    train_ques_embeddings, train_labels, train_par_embeddings = zip(*train)
    test_ques_embeddings, test_labels, test_par_embeddings= zip(*test)

    # Creating test/train datasets
    train_ques_embeddings = np.asarray(train_ques_embeddings)
    dump_embeddings(train_ques_embeddings, os.path.join(base_path, params.files['train_loss']['question_embeddings']))
    train_labels = np.asarray(train_labels)
    dump_embeddings(train_labels, os.path.join(base_path, params.files['train_loss']['question_labels']))
    train_par_embeddings = np.asarray(train_par_embeddings)
    dump_embeddings(train_par_embeddings, os.path.join(base_path, params.files['train_loss']['paragraph_embeddings']))

    test_ques_embeddings = np.asarray(test_ques_embeddings)
    dump_embeddings(test_ques_embeddings, os.path.join(base_path, params.files['test_loss']['question_embeddings']))
    test_labels = np.asarray(test_labels)
    dump_embeddings(test_labels, os.path.join(base_path, params.files['test_loss']['question_labels']))
    test_par_embeddings = np.asarray(test_par_embeddings)
    dump_embeddings(test_par_embeddings, os.path.join(base_path, params.files['test_loss']['paragraph_embeddings']))

    # Creating a subset test set from the test set
    random.shuffle(test)
    subset_test = test[:params.files['splitter']['test_subset_size']]
    subset_test_ques_embeddings, subset_test_labels, subset_test_par_embeddings = zip(*subset_test)
    subset_test_ques_embeddings_file = params.files['subset_file_format']['question_embeddings'].format(params.files['splitter']['test_subset_size'])
    subset_test_par_embeddings_file = params.files['subset_file_format']['paragraph_embeddings'].format(params.files[
                                                                                                             'splitter'][
                                                                                                             'test_subset_size'])
    subset_test_ques_label_file = params.files['subset_file_format']['question_labels'].format(params.files[
                                                                                                             'splitter'][
                                                                                                             'test_subset_size'])

    subset_test_ques_embeddings = np.asarray(subset_test_ques_embeddings)
    dump_embeddings(subset_test_ques_embeddings, os.path.join(base_path, subset_test_ques_embeddings_file))
    subset_test_labels = np.asarray(subset_test_labels)
    dump_embeddings(subset_test_labels, os.path.join(base_path, subset_test_ques_label_file),dtype="int32")
    subset_test_par_embeddings = np.asarray(subset_test_par_embeddings)
    dump_embeddings(subset_test_par_embeddings, os.path.join(base_path, subset_test_par_embeddings_file))


    # update params for new values
    params.files['test_subset_recall']['question_embeddings'] = subset_test_ques_embeddings_file
    params.files['test_subset_recall']['paragraph_embeddings'] = params.files['pre_trained_files']['paragraph_embeddings']
    params.files['test_subset_recall']['question_labels'] = subset_test_ques_label_file

    params.files['test_subset_loss']['question_embeddings'] = subset_test_ques_embeddings_file
    params.files['test_subset_loss']['paragraph_embeddings'] = subset_test_par_embeddings_file

    params.files['splitter']['num_labels'] = num_labels
    params.files['splitter']['train_size'] = train_ques_embeddings.shape[0]
    params.files['splitter']['test_size'] = test_ques_embeddings.shape[0]

    return params


def load_embeddings(infile_to_get):
    with h5py.File(infile_to_get, 'r') as fin:
        document_embeddings = fin['embeddings'][...]
    return document_embeddings

def dump_embeddings(embeddings, outfile_to_dump, dtype="float32"):
    with h5py.File(outfile_to_dump, 'w') as fout:
        ds = fout.create_dataset(
            'embeddings',
            embeddings.shape, dtype=dtype,
            data=embeddings
        )
def analyze_labels(labels_file):
    _labels = pd.read_csv(labels_file)
    analysis = dict()
    df_with_count_of_labels = pd.DataFrame({'count': _labels.groupby(["v"]).size()}).reset_index()

    analysis['df_with_count_of_labels'] = df_with_count_of_labels.copy()
    analysis['mean_of_count'] = df_with_count_of_labels['count'].mean()

    counts_lower_than = 'counts_lower_than_' + str(int(analysis['mean_of_count']))
    df_counts_lower_than = df_with_count_of_labels[df_with_count_of_labels['count'] < int(analysis['mean_of_count'])]
    analysis[counts_lower_than] = (df_counts_lower_than.shape[0], 100 * (df_counts_lower_than .shape[0] /
                                                                         df_with_count_of_labels.shape[0]))

    counts_between= 'counts_between_' + str(int(analysis['mean_of_count'])) + '_' + str(math.ceil(analysis['mean_of_count']))
    df_counts_between= df_with_count_of_labels[(df_with_count_of_labels['count'] >= int(analysis['mean_of_count'])) & (df_with_count_of_labels['count'] <= math.ceil(analysis['mean_of_count']))]

    analysis[counts_between] = (df_counts_between.shape[0], 100 * (df_counts_between.shape[0] /
                                                                         df_with_count_of_labels.shape[0]))

    counts_greater = 'counts_greater_than_' + str(math.ceil(analysis['mean_of_count']))
    df_counts_greater_than = df_with_count_of_labels[df_with_count_of_labels['count'] > math.ceil(analysis['mean_of_count'])]

    analysis[counts_greater ] =(df_counts_greater_than.shape[0], 100 * (df_counts_greater_than.shape[0] /
                                                                         df_with_count_of_labels.shape[0]))

    counts_equal_to_lower = 'counts_equal_to_' + str(int(analysis['mean_of_count']))
    df_counts_equal_to_lower = df_with_count_of_labels[
        df_with_count_of_labels['count'] == int(analysis['mean_of_count'])]
    analysis[counts_equal_to_lower] = (df_counts_equal_to_lower.shape[0], 100 * (df_counts_equal_to_lower.shape[0] /
                                                                          df_counts_between.shape[0]))

    counts_equal_to_higher = 'counts_equal_to_' + str(math.ceil(analysis['mean_of_count']))
    df_counts_equal_to_higher = df_with_count_of_labels[
        df_with_count_of_labels['count'] == math.ceil(analysis['mean_of_count'])]
    analysis[counts_equal_to_higher] = (df_counts_equal_to_higher.shape[0], 100 * (df_counts_equal_to_higher.shape[0] /
                                                                                 df_counts_between.shape[0]))

    #ideal_number_of_paragraphs_for_each_question_for_debug
    analysis['K'] = int(analysis['mean_of_count']) if analysis[counts_equal_to_lower] > analysis[counts_equal_to_higher] else math.ceil(analysis['mean_of_count'])
    return analysis
def dump_mapping_data(data, outfile_to_dump):
    data_df = pd.DataFrame(np.array(data), columns=list("p"))
    data_df.to_csv(outfile_to_dump)


def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    with tf.name_scope('l2_form') as scope:
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True, name='square_sum')
        norm = tf.sqrt(tf.maximum(square_sum, tf.keras.backend.epsilon()), name='norm')
    return norm


def pairwise_cosine_sim(A, B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions
    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point
    """
    with tf.name_scope('pairwise_cosine_sim') as scope:
        A_mag = l2_norm(A, axis=2)
        B_mag = l2_norm(B, axis=2)
        num = tf.keras.backend.batch_dot(A, tf.keras.backend.permute_dimensions(B, (0, 2, 1)))
        den = (A_mag * tf.keras.backend.permute_dimensions(B_mag, (0, 2, 1)))
        dist_mat = num / den
    return dist_mat


def pairwise_euclidean_distances(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pairwise_euclidean_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def calculate_recalls(questions, paragraphs, labels, params, k=None, extract_type=1):
    """
     question [n x d] tensor of n rows with d dimensions
     paragraphs [m x d] tensor of n rows with d dimensions
     params config
     returns:
     loss : scalar value
     """

    recalls = []  # tf.zeros([len(params.recall_top_k), 1], tf.float32)

    # in order to support batch_size feature, we expanded dims for 1
    paragraphs = tf.expand_dims(paragraphs, axis=0)
    labels = tf.expand_dims(labels, axis=0)
    questions = tf.expand_dims(questions, axis=0)
    number_of_questions = tf.to_int64(tf.shape(questions)[1])
    # now question, paragraphs pairwise calculation
    distances = pairwise_cosine_sim(questions, paragraphs)
    for _k in [k] if k is not None else params.recall:
        with tf.name_scope('top_k_{}'.format(_k)) as k_scope:
            # find the top_k paragraphs for each question
            top_k = tf.nn.top_k(distances, k=_k, name='top_k_top_k_{}'.format(_k))

            # is groundtruth label is in these top_k paragraph
            equals = tf.equal(top_k.indices, labels, name='equal_top_k_{}'.format(_k))

            # cast the equals to int32 to count the non zero ones because if it is equal,
            # there is only one 1 for each question among paragraphs,
            # then label is in top k
            casted_equal = tf.cast(equals, dtype=tf.int32, name='casted_equal_top_k_{}'.format(_k))
            final_equals_non_zero = tf.squeeze(
                tf.count_nonzero(casted_equal, axis=2, name='sq_top_k_{}'.format(_k)))


            total_founds_in_k = tf.reduce_sum(final_equals_non_zero)
            recalls.append(total_founds_in_k)

    recalls = tf.stack(recalls)

    return recalls, (recalls / number_of_questions)

def next_batch(begin_indx, batch_size, questions, paragraphs, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    if begin_indx+batch_size <= questions.shape[0]:
        idx = np.arange(begin_indx, begin_indx+batch_size)
    else:
        idx = np.arange(begin_indx, questions.shape[0])
    np.random.shuffle(idx)
    questions = questions[idx]
    labels = labels[idx]
    paragraphs = paragraphs[idx]

    return questions, labels, paragraphs

def get_question_and_paragraph_embeddings(is_cached_question,
                                          is_cached_paragraph,
                                          question_embeddings,
                                          paragraph_embeddings,
                                          params):

    if not is_cached_question:
        _q = load_embeddings(question_embeddings)
    else:
        _q = question_embeddings

    if not is_cached_paragraph:
        _p = load_embeddings(paragraph_embeddings)
    else:
        _p = paragraph_embeddings

    random.seed(params.eval_seed)
    qidx = random.sample(range(_q.shape[0]), params.eval_question_size_for_recall)
    _q = _q[qidx]
    _p = _p[qidx]
    #questions = tf.constant(_q)
    return _q, _p

def closest_distance(batch, paragraph_embeddings, input_type='p2p', score_type='cos'):

    sub_set_ = batch

    if score_type == 'euclidean':
        dist = pairwise_euclidean_distances(sub_set_, paragraph_embeddings)
        top_k = tf.nn.top_k(tf.negative(dist), k=1, name='top_k_{}'.format(1))
        values = tf.reshape(tf.reduce_max(top_k.values, axis=1), shape=[tf.shape(sub_set_)[0], 1])
        scores = tf.negative(values)
        par_indices = tf.reshape(tf.squeeze(top_k.indices), shape=[-1, 1])

    else:
        sub_set_ = tf.expand_dims(sub_set_, axis=0)
        actual_set = tf.expand_dims(paragraph_embeddings, axis=0)
        dist = pairwise_cosine_sim(sub_set_, actual_set)
        if input_type == 'p2p':
            k = 2
        else: #type == 'q2p':
            k = 1
        top_k = tf.nn.top_k(dist, k=k, name='top_k_{}'.format(1))
        values = tf.reshape(tf.reduce_max(top_k.values, axis=2),shape=[tf.shape(sub_set_)[1], 1])
        if input_type == 'p2p':
            par_indices = tf.reshape(tf.squeeze(top_k.indices[:, :, 1]), shape=[-1, 1])
        else:  # type == 'q2p':
            par_indices = tf.reshape(tf.squeeze(top_k.indices), shape=[-1, 1])
        scores = tf.reshape(tf.squeeze(values), shape=[-1,1])

    return scores, par_indices


def question_to_closest_distance(question_embeddings, paragraph_embeddings, batch_size, sess, closest_distance_op, question_tensor, paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]

        batch_distances,par_indices = sess.run(closest_distance_op, feed_dict={
            question_tensor: ques,
            paragraph_tensor: paragraph_embeddings,
        })
        # batch_distances[:,0] -> closest distances
        # batch_distances[:,1] -> indices
        distances = np.append(distances, batch_distances)
    distances = distances.reshape(-1, 1)
    return distances


def question_to_ground_truth_distance(question_embeddings, paragraph_embeddings, batch_size,  sess, euclidean_distance_op, question_tensor,
                        paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]
        pars = paragraph_embeddings[start:end]

        batch_distances= sess.run(euclidean_distance_op, feed_dict={
            question_tensor: ques,
            paragraph_tensor: pars,
        })
        distances = np.append(distances, batch_distances)
    distances = distances.reshape(-1, 1)
    return distances

def question_to_random_paragraph_distance(question_embeddings, paragraph_embeddings, batch_size,  sess, euclidean_distance_op, question_tensor,
                        paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    np.random.shuffle(paragraph_embeddings)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]
        pars = paragraph_embeddings[start:end]
        np.random.shuffle(pars)
        batch_distances= sess.run(euclidean_distance_op, feed_dict={
            question_tensor: ques,
            paragraph_tensor: pars,
        })
        distances = np.append(distances, batch_distances)
    distances = distances.reshape(-1, 1)
    return distances

def list_slice(tensor, indices, axis):
    """
    Args
    ----
    tensor (Tensor) : input tensor to slice
    indices ( [int] ) : list of indices of where to perform slices
    axis (int) : the axis to perform the slice on
    """

    slices = []

    ## Set the shape of the output tensor.
    # Set any unknown dimensions to -1, so that reshape can infer it correctly.
    # Set the dimension in the slice direction to be 1, so that overall dimensions are preserved during the operation
    shape = tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    nd = len(shape)

    for i in indices:
        _slice = [slice(None)]*nd
        _slice[axis] = slice(i,i+1)
        slices.append(tf.reshape(tensor[_slice], shape))

    return tf.concat(slices, axis=axis)

def get_variable_name_as_str(variable):
    return [k for k, v in locals().items() if v == variable][0]