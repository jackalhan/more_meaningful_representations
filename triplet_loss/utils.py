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

def train_test_splitter_2(embeddings_file, labels_file, train_split_rate, debug=False):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
    """
    _e = embeddings_file.rpartition(os.path.sep)
    path_e = _e[0]
    splitted_train_embed_file = os.path.join(path_e, 'splitted_train' + _e[2].replace('train',''))
    splitted_test_embed_file = os.path.join(path_e, 'splitted_test' + _e[2].replace('train',''))

    _l = labels_file.rpartition(os.path.sep)
    path_l = _l[0]
    splitted_train_label_file = os.path.join(path_l, 'splitted_train' + _l[2].replace('train', ''))
    splitted_test_label_file = os.path.join(path_l, 'splitted_test' + _l[2].replace('train', ''))

    _labels = pd.read_csv(labels_file)
    embeddings = load_embeddings(embeddings_file)
    labels = _labels['v'].tolist()
    num_labels = len(set(labels))
    if debug:
        num_labels = 1000
    #in order to have all the labels in the training set, we need to split them accordingly:
    train_labels, test_labels= list(), list()
    train_embeddings, test_embeddings = list(), list()
    #counter = 0
    for i in range(num_labels):
        locations = [_ for _, x in enumerate(labels) if x == i]
        occur = len(locations)
        print(10 * '*')
        print('For p {}, we have -> {} qs ---> {}'.format(i, occur, locations))
        for_local_train_size = math.ceil(occur * train_split_rate)
        for_local_train_locations = locations[0:for_local_train_size]
        for_local_train_labels = list()
        for_local_train_embeddings = list()

        for _l in for_local_train_locations:
            for_local_train_labels.append(i)
            for_local_train_embeddings.append(embeddings[_l])
            #counter += 1
        train_labels.extend(for_local_train_labels)
        train_embeddings.extend(for_local_train_embeddings)
        print('Train Size {} ---> {}'.format(for_local_train_size, for_local_train_locations))


        for_local_test_locations = locations[for_local_train_size:]
        for_local_test_size = len(for_local_test_locations)
        for_local_test_labels = list()
        for_local_test_embeddings = list()
        for _l in for_local_test_locations:
            for_local_test_labels.append(i)
            for_local_test_embeddings.append(embeddings[_l])
            #counter += 1
        test_labels.extend(for_local_test_labels)
        test_embeddings.extend(for_local_test_embeddings)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    if not debug:
        assert num_labels == len(set(train_labels)), "Actual Num of Labels: {} vs Train Num of Labels {}".format(num_labels, len(set(train_labels)))
        # assert num_labels == len(set(test_labels)), "Actual Num of Labels: {} vs Test Num of Labels {}".format(num_labels,
        #                                                                                                          len(set(
        #                                                                                                         test_labels)))
    train_embeddings_label = list(zip(train_embeddings, train_labels))
    test_embeddings_label = list(zip(test_embeddings, test_labels))

    random.shuffle(train_embeddings_label)
    random.shuffle(test_embeddings_label)
    train_embeddings, train_labels = zip(*train_embeddings_label)
    test_embeddings, test_labels = zip(*test_embeddings_label)

    train_embeddings = np.asarray(train_embeddings)
    dump_embeddings(train_embeddings, splitted_train_embed_file)
    dump_mapping_data(train_labels, splitted_train_label_file)

    test_embeddings = np.asarray(test_embeddings)
    dump_embeddings(test_embeddings, splitted_test_embed_file)
    dump_mapping_data(test_labels, splitted_test_label_file)

    file_paths = {}
    file_paths['train_embeddings'] = splitted_train_embed_file
    file_paths['train_labels'] = splitted_train_label_file
    file_paths['train_size'] = train_embeddings.shape[0]

    file_paths['test_embeddings'] = splitted_test_embed_file
    file_paths['test_labels'] = splitted_test_label_file
    file_paths['eval_size'] = test_embeddings.shape[0]
    file_paths['data_dim'] = test_embeddings.shape[1]
    file_paths['num_labels'] = num_labels
    return file_paths

def train_test_splitter(embeddings_file, labels_file, K):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
            K: labels that has K samples
    """
    _e = embeddings_file.rpartition(os.path.sep)
    path_e = _e[0]
    splitted_train_embed_file = os.path.join(path_e, 'splitted_train' + _e[2].replace('train', ''))
    splitted_test_embed_file = os.path.join(path_e, 'splitted_test' + _e[2].replace('train', ''))

    _l = labels_file.rpartition(os.path.sep)
    path_l = _l[0]
    splitted_train_label_file = os.path.join(path_l, 'splitted_train' + _l[2].replace('train', ''))
    splitted_test_label_file = os.path.join(path_l, 'splitted_test' + _l[2].replace('train', ''))

    _labels = pd.read_csv(labels_file)
    _count_of_labels = pd.DataFrame({'count': _labels.groupby(["v"]).size()}).reset_index()
    _count_of_labels = _count_of_labels['count'].values
    def assign_counts_to_each_row(row):
        return int(_count_of_labels[row['v']])

    _labels['K'] = _labels.apply(assign_counts_to_each_row, axis=1)
    _embeddings = load_embeddings(embeddings_file)
    _embeddings = np.concatenate((_embeddings, _labels[['v', 'K']]), 1)
    _embeddings = _embeddings[_embeddings[:, _embeddings.shape[1] - 1] == K]

    embeddings, labels = _embeddings[:, :_embeddings.shape[1] - 2], _embeddings[:,_embeddings.shape[1] - 2:
                                                                                  _embeddings.shape[
                                                                               1] - 1].astype(np.int32)

    # random.shuffle(embeddings)
    # train_size = int(embeddings.shape[0] * train_split_rate)
    # train_embeddings_label = embeddings[:train_size]
    # test_embeddings_label = embeddings[train_size:]
    # train_embeddings, train_labels = train_embeddings_label[:, :embeddings.shape[1] - 2], train_embeddings_label[:,
    #                                                                                       embeddings.shape[1] - 2:
    #                                                                                       embeddings.shape[
    #                                                                                           1] - 1].astype(np.int32)
    # test_embeddings, test_labels = test_embeddings_label[:, :embeddings.shape[1] - 2], train_embeddings_label[:,
    #                                                                                    embeddings.shape[1] - 2:
    #                                                                                    embeddings.shape[1] - 1].astype(
    #     np.int32)
    #
    # dump_embeddings(train_embeddings, splitted_train_embed_file)
    # dump_mapping_data(train_labels, splitted_train_label_file)
    #
    # dump_embeddings(test_embeddings, splitted_test_embed_file)
    # dump_mapping_data(test_labels, splitted_test_label_file)

    #
    #labels = _labels['v'].tolist()
    num_labels = np.unique(labels[:,:]).shape[0]
    train_labels, test_labels = list(), list()
    train_embeddings, test_embeddings = list(), list()
    # counter = 0
    for i in np.unique(labels[:,:]):
        locations = [_ for _, x in enumerate(labels) if x == i]
        shuffle(locations)
        occur = len(locations)
        print(10 * '*')
        print('For p {}, we have -> {} qs ---> {}'.format(i, occur, locations))
        for_local_train_size = K-1
        for_local_train_locations = locations[0:for_local_train_size]
        for_local_train_labels = list()
        for_local_train_embeddings = list()

        for _l in for_local_train_locations:
            for_local_train_labels.append(i)
            for_local_train_embeddings.append(embeddings[_l])
            #counter += 1
        train_labels.extend(for_local_train_labels)
        train_embeddings.extend(for_local_train_embeddings)
        print('Train Size {} ---> {}'.format(for_local_train_size, for_local_train_locations))


        for_local_test_locations = locations[for_local_train_size:]
        for_local_test_size = len(for_local_test_locations)
        for_local_test_labels = list()
        for_local_test_embeddings = list()
        for _l in for_local_test_locations:
            for_local_test_labels.append(i)
            for_local_test_embeddings.append(embeddings[_l])
            #counter += 1
        test_labels.extend(for_local_test_labels)
        test_embeddings.extend(for_local_test_embeddings)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    train_embeddings_label = list(zip(train_embeddings, train_labels))
    test_embeddings_label = list(zip(test_embeddings, test_labels))

    random.shuffle(train_embeddings_label)
    random.shuffle(test_embeddings_label)
    train_embeddings, train_labels = zip(*train_embeddings_label)
    test_embeddings, test_labels = zip(*test_embeddings_label)

    train_embeddings = np.asarray(train_embeddings)
    dump_embeddings(train_embeddings, splitted_train_embed_file)
    dump_mapping_data(train_labels, splitted_train_label_file)

    test_embeddings = np.asarray(test_embeddings)
    dump_embeddings(test_embeddings, splitted_test_embed_file)
    dump_mapping_data(test_labels, splitted_test_label_file)

    file_paths = {}
    file_paths['train_embeddings'] = splitted_train_embed_file
    file_paths['train_labels'] = splitted_train_label_file
    file_paths['train_size'] = train_embeddings.shape[0]

    file_paths['test_embeddings'] = splitted_test_embed_file
    file_paths['test_labels'] = splitted_test_label_file
    file_paths['eval_size'] = test_embeddings.shape[0]
    file_paths['data_dim'] = test_embeddings.shape[1]
    file_paths['num_labels'] = num_labels
    return file_paths


def load_embeddings(infile_to_get):
    with h5py.File(infile_to_get, 'r') as fin:
        document_embeddings = fin['embeddings'][...]
    return document_embeddings

def dump_embeddings(embeddings, outfile_to_dump):
    with h5py.File(outfile_to_dump, 'w') as fout:
        ds = fout.create_dataset(
            'embeddings',
            embeddings.shape, dtype='float32',
            data=embeddings
        )
def analyze_labes(labels_file):
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
    data_df = pd.DataFrame(np.array(data), columns=list("v"))
    data_df.to_csv(outfile_to_dump)