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

def train_test_splitter(question_embeddings_file, paragraph_embeddings_file, labels_file, train_split_rate, limit_data=None):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
    """
    _q = question_embeddings_file.rpartition(os.path.sep)
    path_q = _q[0]
    splitted_train_ques_embed_file = os.path.join(path_q, 'splitted_train' + _q[2].replace('train',''))
    splitted_test_ques_embed_file = os.path.join(path_q, 'splitted_test' + _q[2].replace('train',''))

    _l = labels_file.rpartition(os.path.sep)
    path_l = _l[0]
    splitted_train_label_file = os.path.join(path_l, 'splitted_train' + _l[2].replace('train', ''))
    splitted_test_label_file = os.path.join(path_l, 'splitted_test' + _l[2].replace('train', ''))

    _p = paragraph_embeddings_file.rpartition(os.path.sep)
    path_p = _p[0]
    splitted_train_par_embed_file = os.path.join(path_p, 'splitted_train' + _p[2].replace('train', ''))
    splitted_test_par_embed_file = os.path.join(path_q, 'splitted_test' + _p[2].replace('train', ''))

    _labels = pd.read_csv(labels_file)
    question_embeddings = load_embeddings(question_embeddings_file)
    paragraph_embeddings = load_embeddings(paragraph_embeddings_file)
    labels = _labels['v'].tolist()
    num_labels = len(set(labels))
    if limit_data is not None:
        num_labels = limit_data
    labels_as_list = list(range(num_labels))
    shuffle(labels_as_list)
    #in order to have all the labels in the training set, we need to split them accordingly:
    train_labels, test_labels= list(), list()
    train_ques_embeddings, test_ques_embeddings = list(), list()
    train_par_embeddings, test_par_embeddings = list(), list()
    for i in labels_as_list:
        locations = [_ for _, x in enumerate(labels) if x == i]
        shuffle(locations)
        occur = len(locations)
        print(10 * '*')
        print('For p {}, we have -> {} qs ---> {}'.format(i, occur, locations))
        for_local_train_size = math.ceil(occur * train_split_rate)
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
            #counter += 1
        test_labels.extend(for_local_test_labels)
        test_ques_embeddings.extend(for_local_test_ques_embeddings)
        test_par_embeddings.extend(for_local_test_par_embeddings)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    if limit_data is None:
        assert num_labels == len(set(train_labels)), "Actual Num of Labels: {} vs Train Num of Labels {}".format(num_labels, len(set(train_labels)))

    train_embeddings_label = list(zip(train_ques_embeddings, train_labels, train_par_embeddings))
    test_embeddings_label = list(zip(test_par_embeddings, test_labels, test_par_embeddings))

    random.shuffle(train_embeddings_label)
    random.shuffle(test_embeddings_label)

    train_ques_embeddings, train_labels, train_par_embeddings = zip(*train_embeddings_label)
    test_ques_embeddings, test_labels, test_par_embeddings= zip(*test_embeddings_label)

    train_ques_embeddings = np.asarray(train_ques_embeddings)
    dump_embeddings(train_ques_embeddings, splitted_train_ques_embed_file)
    train_par_embeddings = np.asarray(train_par_embeddings)
    dump_embeddings(train_par_embeddings, splitted_train_par_embed_file)
    dump_mapping_data(train_labels, splitted_train_label_file)

    test_ques_embeddings = np.asarray(test_ques_embeddings)
    dump_embeddings(test_ques_embeddings, splitted_test_ques_embed_file)
    test_par_embeddings = np.asarray(test_par_embeddings)
    dump_embeddings(test_par_embeddings, splitted_test_par_embed_file)
    dump_mapping_data(test_labels, splitted_test_label_file)

    file_paths = {}
    file_paths['train_question_embeddings'] = splitted_train_ques_embed_file
    file_paths['train_paragraph_embeddings'] = splitted_train_par_embed_file
    file_paths['train_paragraph_labels'] = splitted_train_label_file
    file_paths['train_question_size'] = train_ques_embeddings.shape[0]

    file_paths['test_question_embeddings'] = splitted_test_ques_embed_file
    file_paths['test_paragraph_embeddings'] = splitted_test_par_embed_file
    file_paths['test_paragraph_labels'] = splitted_test_label_file

    file_paths['eval_question_size'] = test_ques_embeddings.shape[0]
    file_paths['data_dim'] = train_ques_embeddings.shape[1]
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