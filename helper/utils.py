"""General utility functions"""

import json
import logging
import pandas as pd
import h5py
import random
import numpy as np
import math
import os
from random import shuffle, seed
import tensorflow as tf
import shutil
from tqdm import tqdm
import pickle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import shelve
import spacy
import datetime
from collections import defaultdict, Counter
nlp = spacy.blank("en")
#nlp_s = spacy.load('en')
encoding="utf-8"
#tokenize = lambda doc: [token.text for token in nlp(doc)]
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

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

def load_word_embeddings(path, vocab, embedding_size):
    embeddings = {}
    vocab_size = len(vocab)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors

    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    num_loaded = 0
    for w, i in vocab.items():
        v = embeddings.get(w)
        if v is not None and i < vocab_size:
            embedding_matrix[i] = v
            num_loaded += 1
        else:
            print('Token {} with {} id is not found'.format(w,v))
    print('Successfully loaded pretrained embeddings for {}/{} words'.format(num_loaded, vocab_size))
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix

def vocabulary_processor(tokenized_docs):
    indx_to_voc = {0: '<PAD>'} #, 1: '<S>', 2:'</S>',3:'<UNK>'
    voc_to_indx = {'<PAD>':0} # , '<S>':1, '</S>':2,'<UNK>':3
    offset = 1 # 4
    for tokens in tokenized_docs:
        for token in tokens:
            if token not in voc_to_indx:
                voc_to_indx[token] = offset
                indx_to_voc[offset] = token
                offset +=1
    print('{} vocs are created'.format(offset))
    return indx_to_voc, voc_to_indx

def fit_vocab_to_documents(tokenized_docs, voc_to_indx):
    document_lists = []
    for doc_indx, tokens in enumerate(tokenized_docs):
        token_list = []
        #token_list.append(voc_to_indx['<S>'])
        for token in tokens:
            try:
                token_list.append(voc_to_indx[token])
            except:
                print('{} token in Doc {} could not found'.format(token,doc_indx))
        #token_list.append(voc_to_indx['</S>'])
        document_lists.append(token_list)
    document_lists = np.array(document_lists)
    return document_lists



def define_pre_executions(params, json_path, base_data_path):
    model_save_path = os.path.join(params.executor['model_dir'], params.executor['save_dir'],
                                   create_execution_name(params))
    if params.executor['analyze_labels']:
        analysis = analyze_labels(params.files['pre_trained_files']['question_labels'])
        print(analysis)

    else:
        if params.executor["split_train_set"]:
            params = train_test_splitter(params, base_data_path)

            params.save(json_path)
            print('Done with splitting')
        if not params.executor["is_prediction"]:
            if not params.executor["is_training_resume"]:
                try:
                    shutil.rmtree(model_save_path)
                except:
                    pass
        else:
            model_save_path = os.path.join(params.executor['model_dir'], params.executor['save_dir'], params.executor['pre_trained_model_name'])
    return params, model_save_path


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


def train_test_splitter_deprecated(params, base_path):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
    """
    pre_trained_question_embeddings_file = os.path.join(base_path,
                                                        params.files['pre_trained_files']['question_embeddings'])
    pre_trained_paragraph_embeddings_file = os.path.join(base_path,
                                                         params.files['pre_trained_files']['paragraph_embeddings'])
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
    seed(params.model['seed'])
    shuffle(labels_as_list)
    # in order to have all the labels in the training set, we need to split them accordingly:
    train_labels, test_labels = list(), list()
    train_ques_indices, train_ques_embeddings, train_par_embeddings = list(), list(), list()
    test_ques_indices, test_ques_embeddings, test_par_embeddings = list(), list(), list()
    recall_paragraph_embeddings = list()
    for par_order, par_indx in enumerate(labels_as_list):
        locations = [_ for _, x in enumerate(pre_trained_question_labels) if x == par_indx]
        seed(params.model['seed'])
        shuffle(locations)
        occur = len(locations)
        print(10 * '*')
        print('For p: {}, we have -> {} qs ---> {}'.format(par_indx, occur, locations))
        for_local_train_size = math.ceil(occur * params.files['splitter']['train_split_rate'])
        for_local_train_locations = locations[0:for_local_train_size]
        for_local_train_labels = list()
        for_local_train_ques_embeddings = list()
        for_local_train_par_embeddings = list()
        for_local_train_ques_indices = list()
        recall_paragraph_embeddings.append(paragraph_embeddings[par_indx])
        for _l in for_local_train_locations:
            for_local_train_labels.append(par_order)
            for_local_train_ques_embeddings.append(question_embeddings[_l])
            for_local_train_par_embeddings.append(paragraph_embeddings[par_indx])
            for_local_train_ques_indices.append(_l)
        train_labels.extend(for_local_train_labels)
        train_ques_embeddings.extend(for_local_train_ques_embeddings)
        train_par_embeddings.extend(for_local_train_par_embeddings)
        train_ques_indices.extend(for_local_train_ques_indices)
        print('Train Size {} ---> {}'.format(for_local_train_size, for_local_train_locations))

        for_local_test_locations = locations[for_local_train_size:]
        for_local_test_size = len(for_local_test_locations)
        for_local_test_labels = list()
        for_local_test_ques_embeddings = list()
        for_local_test_par_embeddings = list()
        for_local_test_ques_indices = list()
        for _l in for_local_test_locations:
            for_local_test_labels.append(par_order)
            for_local_test_ques_embeddings.append(question_embeddings[_l])
            for_local_test_par_embeddings.append(paragraph_embeddings[par_indx])
            for_local_test_ques_indices.append(_l)
        test_labels.extend(for_local_test_labels)
        test_ques_embeddings.extend(for_local_test_ques_embeddings)
        test_par_embeddings.extend(for_local_test_par_embeddings)
        test_ques_indices.extend(for_local_test_ques_indices)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    # assert num_labels == len(set(test_labels)), "Actual Num of Labels: {} vs Test Num of Labels {}".format(num_labels, len(set(test_labels)))

    train = list(zip(train_ques_embeddings, train_labels, train_par_embeddings, train_ques_indices))
    test = list(zip(test_ques_embeddings, test_labels, test_par_embeddings, test_ques_indices))
    random.seed(params.model['seed'])
    random.shuffle(train)
    random.seed(params.model['seed'])
    random.shuffle(test)
    train_ques_embeddings, train_labels, train_par_embeddings, train_ques_indices = zip(*train)
    test_ques_embeddings, test_labels, test_par_embeddings, test_ques_indices = zip(*test)

    # Creating sample paragraphs:

    recall_paragraph_embedding_file = 'recall_' + params.files['pre_trained_files']['paragraph_embeddings']
    recall_paragraph_embeddings = np.asarray(recall_paragraph_embeddings)

    dump_embeddings(recall_paragraph_embeddings,
                    os.path.join(base_path, recall_paragraph_embedding_file))
    # Creating test/train datasets
    train_ques_embeddings = np.asarray(train_ques_embeddings)
    dump_embeddings(train_ques_embeddings, os.path.join(base_path, params.files['train_loss']['question_embeddings']))
    train_labels = np.asarray(train_labels)
    dump_embeddings(train_labels, os.path.join(base_path, params.files['train_loss']['question_labels']))
    train_par_embeddings = np.asarray(train_par_embeddings)
    dump_embeddings(train_par_embeddings, os.path.join(base_path, params.files['train_loss']['paragraph_embeddings']))
    train_ques_indices = np.asarray(train_ques_indices)
    dump_embeddings(train_ques_indices, os.path.join(base_path, params.files['train_loss']['question_idx']))

    test_ques_embeddings = np.asarray(test_ques_embeddings)
    dump_embeddings(test_ques_embeddings, os.path.join(base_path, params.files['test_loss']['question_embeddings']))
    test_labels = np.asarray(test_labels)
    dump_embeddings(test_labels, os.path.join(base_path, params.files['test_loss']['question_labels']))
    test_par_embeddings = np.asarray(test_par_embeddings)
    dump_embeddings(test_par_embeddings, os.path.join(base_path, params.files['test_loss']['paragraph_embeddings']))
    test_ques_indices = np.asarray(test_ques_indices)
    dump_embeddings(test_ques_indices, os.path.join(base_path, params.files['test_loss']['question_idx']))

    # Creating a subset test set from the test set
    random.seed(params.model['seed'])
    random.shuffle(test)
    if params.executor['limit_paragraph_size'] is not None:
        number_of_files_for_test = range(1, params.files['splitter']['test_subset_size'] + 1)
    else:
        number_of_files_for_test = [params.files['splitter']['test_subset_size']]
    for _size in number_of_files_for_test:
        subset_test = test[:_size]
        subset_test_ques_embeddings, subset_test_labels, subset_test_par_embeddings, subset_test_ques_indices = zip(*subset_test)
        subset_test_ques_embeddings_file = params.files['subset_file_format']['question_embeddings'].format('test', _size)
        subset_test_par_embeddings_file = params.files['subset_file_format']['paragraph_embeddings'].format('test', _size)
        subset_test_ques_label_file = params.files['subset_file_format']['question_labels'].format('test',_size)
        subset_test_ques_idx_file = params.files['subset_file_format']['question_idx'].format('test',_size)

        subset_test_ques_embeddings = np.asarray(subset_test_ques_embeddings)
        dump_embeddings(subset_test_ques_embeddings, os.path.join(base_path, subset_test_ques_embeddings_file))
        subset_test_labels = np.asarray(subset_test_labels)
        dump_embeddings(subset_test_labels, os.path.join(base_path, subset_test_ques_label_file), dtype="int32")
        subset_test_par_embeddings = np.asarray(subset_test_par_embeddings)
        dump_embeddings(subset_test_par_embeddings, os.path.join(base_path, subset_test_par_embeddings_file))
        subset_test_ques_indices = np.asarray(subset_test_ques_indices)
        dump_embeddings(subset_test_ques_indices, os.path.join(base_path, subset_test_ques_idx_file))

    random.seed(params.model['seed'])
    random.shuffle(train)
    if params.executor['limit_paragraph_size'] is not None:
        number_of_files_for_train = range(1, params.files['splitter']['train_subset_size'] + 1)
    else:
        number_of_files_for_train = [params.files['splitter']['train_subset_size']]

    for _size in number_of_files_for_train:
        subset_train = train[:_size]
        subset_train_ques_embeddings, subset_train_labels, subset_train_par_embeddings, subset_train_ques_indices = zip(*subset_train)
        subset_train_ques_embeddings_file = params.files['subset_file_format']['question_embeddings'].format('train_recall',_size)
        subset_train_par_embeddings_file = params.files['subset_file_format']['paragraph_embeddings'].format('train_recall', _size)
        subset_train_ques_label_file = params.files['subset_file_format']['question_labels'].format('train_recall', _size)
        subset_train_ques_idx_file = params.files['subset_file_format']['question_idx'].format('train_recall',_size)

        subset_train_ques_embeddings = np.asarray(subset_train_ques_embeddings)
        dump_embeddings(subset_train_ques_embeddings, os.path.join(base_path, subset_train_ques_embeddings_file))
        subset_train_labels = np.asarray(subset_train_labels)
        dump_embeddings(subset_train_labels, os.path.join(base_path, subset_train_ques_label_file), dtype="int32")
        subset_train_par_embeddings = np.asarray(subset_train_par_embeddings)
        dump_embeddings(subset_train_par_embeddings, os.path.join(base_path, subset_train_par_embeddings_file))
        subset_train_ques_indices = np.asarray(subset_train_ques_indices)
        dump_embeddings(subset_train_ques_indices, os.path.join(base_path, subset_train_ques_idx_file))

    # update params for new values
    params.files['test_subset_recall']['question_embeddings'] = subset_test_ques_embeddings_file
    params.files['test_subset_recall']['paragraph_embeddings'] = subset_test_par_embeddings_file
    params.files['test_subset_recall']['question_labels'] = subset_test_ques_label_file
    params.files['test_subset_recall']['question_idx'] = subset_test_ques_idx_file

    params.files['train_subset_recall']['question_embeddings'] = subset_train_ques_embeddings_file
    params.files['train_subset_recall']['paragraph_embeddings'] = subset_train_par_embeddings_file
    params.files['train_subset_recall']['question_labels'] = subset_train_ques_label_file
    params.files['train_subset_recall']['question_idx'] = subset_train_ques_idx_file

    params.files['test_subset_loss']['question_embeddings'] = subset_test_ques_embeddings_file
    params.files['test_subset_loss']['paragraph_embeddings'] = subset_test_par_embeddings_file

    params.files['splitter']['num_labels'] = num_labels
    params.files['splitter']['train_size'] = train_ques_embeddings.shape[0]
    params.files['splitter']['test_size'] = test_ques_embeddings.shape[0]

    return params

def get_file_key_names_for_execution(params):
    KN_FILE_NAMES = {}
    if params.executor['source'].lower().startswith('qu'):
        #key names
        KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"] = 'question_embeddings'
        KN_FILE_NAMES["KN_SOURCE_LABELS"]= 'question_labels'
        KN_FILE_NAMES["KN_SOURCE_IDX"] = 'question_idx'
        KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]='paragraph_embeddings'
        KN_FILE_NAMES["DIR"] = 'question_training'
    else:
        # key names
        KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"] = 'paragraph_embeddings'
        KN_FILE_NAMES["KN_SOURCE_LABELS"] = 'paragraph_labels'
        KN_FILE_NAMES["KN_SOURCE_IDX"] = 'paragraph_idx'
        KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"] = 'question_embeddings'
        KN_FILE_NAMES["DIR"] = 'paragraph_training'

    return KN_FILE_NAMES

def train_test_splitter(params, base_path):
    """ Read the embedding file with its labels and split it train - test
        Args:
            embeddings_file: embeddings_file path
            label_file: labels file path
            train_split_rate: rate of the train dataset from whole records
    """
    pre_trained_question_embeddings_file = os.path.join(base_path,
                                                        params.files['pre_trained_files']['question_embeddings'])
    pre_trained_paragraph_embeddings_file = os.path.join(base_path,
                                                         params.files['pre_trained_files']['paragraph_embeddings'])
    pre_trained_question_labels_file = os.path.join(base_path, params.files['pre_trained_files']['question_labels'])

    df_pre_trained_question_labels = pd.read_csv(pre_trained_question_labels_file)
    pre_trained_question_labels = df_pre_trained_question_labels['p'].tolist()
    pre_trained_paragraph_labels = df_pre_trained_question_labels['q'].tolist()
    question_embeddings = load_embeddings(pre_trained_question_embeddings_file)
    paragraph_embeddings = load_embeddings(pre_trained_paragraph_embeddings_file)

    KN_FILE_NAMES = get_file_key_names_for_execution(params)
    if params.executor['source'].lower().startswith('qu'):
        target = 'paragraph'
        source_label_list = pre_trained_question_labels
        source_embeddings = question_embeddings
        target_embeddings = paragraph_embeddings
    else:
        target = 'question'
        source_label_list = pre_trained_paragraph_labels
        source_embeddings = paragraph_embeddings
        target_embeddings = question_embeddings

    base_path = create_dir(os.path.join(base_path, KN_FILE_NAMES['DIR']))
    num_labels = len(set(source_label_list))
    if params.executor['limit_source_size'] is not None:
        # just grab the items in the source side within the size of the source size that are limited with this parameter
        num_labels = params.executor['limit_source_size']
        number_of_files_for_train = range(1, params.files['splitter']['train_subset_size'] + 1)
        number_of_files_for_test = range(1, params.files['splitter']['test_subset_size'] + 1)
    else:
        number_of_files_for_train = [params.files['splitter']['train_subset_size']]
        number_of_files_for_test = [params.files['splitter']['test_subset_size']]

    labels_as_list = list(range(num_labels))
    seed(params.model['seed'])
    shuffle(labels_as_list)
    # in order to have all the labels in the training set, we need to split them accordingly:
    train_labels, test_labels = list(), list()
    train_source_indices, train_source_embeddings, train_target_embeddings = list(), list(), list()
    test_source_indices, test_source_embeddings, test_target_embeddings = list(), list(), list()
    recall_target_embeddings = list()
    for order, indx in enumerate(labels_as_list):
        #indexes from source list where each of target equals to particular value.
        #sample: par_indx = 3, take all the questions's locationswhere their par_id =3
        #sample: que_indx = 5000, take all the paragraphs locations where their que_id = 5000 (paragraph has only 1 question that is 5000)
        #therefore, can we get all paragraphs locations that are in the group of same paragraph in which que_id = 5000
        if not params.executor['source'].lower().startswith('qu'):
            paragraph = pre_trained_question_labels[indx]
            locations = [paragraph for _, x in enumerate(pre_trained_question_labels) if x == paragraph]
        else:
            locations = [_ for _, x in enumerate(source_label_list) if x == indx]
        seed(params.model['seed'])
        shuffle(locations)
        occur = len(locations)
        print(10 * '*')
        print('For target({}) : {}, we have -> {} sources({}) ---> {}'.format(target, indx, occur, params.executor['source'].lower() ,locations))
        for_local_train_size = math.ceil(occur * params.files['splitter']['train_split_rate'])
        for_local_train_locations = locations[0:for_local_train_size]
        for_local_train_labels = list()
        for_local_train_source_embeddings = list()
        for_local_train_target_embeddings = list()
        for_local_train_source_indices = list()
        recall_target_embeddings.append(target_embeddings[indx])
        for _l in for_local_train_locations:
            for_local_train_labels.append(order)
            for_local_train_source_embeddings.append(source_embeddings[_l])
            for_local_train_target_embeddings.append(target_embeddings[indx])
            for_local_train_source_indices.append(_l)
        train_labels.extend(for_local_train_labels)
        train_source_embeddings.extend(for_local_train_source_embeddings)
        train_target_embeddings.extend(for_local_train_target_embeddings)
        train_source_indices.extend(for_local_train_source_indices)
        print('Train Size {} ---> {}'.format(for_local_train_size, for_local_train_locations))

        for_local_test_locations = locations[for_local_train_size:]
        for_local_test_size = len(for_local_test_locations)
        for_local_test_labels = list()
        for_local_test_source_embeddings = list()
        for_local_test_target_embeddings = list()
        for_local_test_source_indices = list()
        for _l in for_local_test_locations:
            for_local_test_labels.append(order)
            for_local_test_source_embeddings.append(source_embeddings[_l])
            for_local_test_target_embeddings.append(target_embeddings[indx])
            for_local_test_source_indices.append(_l)
        test_labels.extend(for_local_test_labels)
        test_source_embeddings.extend(for_local_test_source_embeddings)
        test_target_embeddings.extend(for_local_test_target_embeddings)
        test_source_indices.extend(for_local_test_source_indices)
        print('Test Size {} ---> {}'.format(for_local_test_size, for_local_test_locations))

    # assert num_labels == len(set(test_labels)), "Actual Num of Labels: {} vs Test Num of Labels {}".format(num_labels, len(set(test_labels)))

    train = list(zip(train_source_embeddings, train_labels, train_target_embeddings, train_source_indices))
    test = list(zip(test_source_embeddings, test_labels, test_target_embeddings, test_source_indices))
    random.seed(params.model['seed'])
    random.shuffle(train)
    random.seed(params.model['seed'])
    random.shuffle(test)
    train_source_embeddings, train_labels, train_target_embeddings, train_source_indices = zip(*train)
    test_source_embeddings, test_labels, test_target_embeddings, test_source_indices = zip(*test)

    # Creating sample paragraphs:

    recall_target_embedding_file = 'recall_' + params.files['pre_trained_files'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]]
    recall_target_embeddings = np.asarray(recall_target_embeddings)

    dump_embeddings(recall_target_embeddings,
                    os.path.join(base_path, recall_target_embedding_file))

    # Creating test/train datasets
    train_source_embeddings = np.asarray(train_source_embeddings)
    dump_embeddings(train_source_embeddings, os.path.join(base_path, params.files['train_loss'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]]))
    train_labels = np.asarray(train_labels)
    dump_embeddings(train_labels, os.path.join(base_path, params.files['train_loss'][KN_FILE_NAMES["KN_SOURCE_LABELS"]]))
    train_target_embeddings = np.asarray(train_target_embeddings)
    dump_embeddings(train_target_embeddings, os.path.join(base_path, params.files['train_loss'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]]))
    train_source_indices = np.asarray(train_source_indices)
    dump_embeddings(train_source_indices, os.path.join(base_path, params.files['train_loss'][KN_FILE_NAMES["KN_SOURCE_IDX"]]))

    test_source_embeddings = np.asarray(test_source_embeddings)
    dump_embeddings(test_source_embeddings, os.path.join(base_path, params.files['test_loss'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]]))
    test_labels = np.asarray(test_labels)
    dump_embeddings(test_labels, os.path.join(base_path, params.files['test_loss'][KN_FILE_NAMES["KN_SOURCE_LABELS"]]))
    test_target_embeddings = np.asarray(test_target_embeddings)
    dump_embeddings(test_target_embeddings, os.path.join(base_path, params.files['test_loss'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]]))
    test_source_indices = np.asarray(test_source_indices)
    dump_embeddings(test_source_indices, os.path.join(base_path, params.files['test_loss'][KN_FILE_NAMES["KN_SOURCE_IDX"]]))

    # Creating a subset test set from the test set
    random.seed(params.model['seed'])
    random.shuffle(test)

    for _size in number_of_files_for_test:
        subset_test = test[:_size]
        subset_test_source_embeddings, subset_test_labels, subset_test_target_embeddings, subset_test_source_indices = zip(*subset_test)
        subset_test_source_embeddings_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]].format('test', _size)
        subset_test_target_embeddings_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]].format('test', _size)
        subset_test_source_label_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_LABELS"]].format('test',_size)
        subset_test_source_idx_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_IDX"]].format('test',_size)

        subset_test_source_embeddings = np.asarray(subset_test_source_embeddings)
        dump_embeddings(subset_test_source_embeddings, os.path.join(base_path, subset_test_source_embeddings_file))
        subset_test_labels = np.asarray(subset_test_labels)
        dump_embeddings(subset_test_labels, os.path.join(base_path, subset_test_source_label_file), dtype="int32")
        subset_test_target_embeddings = np.asarray(subset_test_target_embeddings)
        dump_embeddings(subset_test_target_embeddings, os.path.join(base_path, subset_test_target_embeddings_file))
        subset_test_source_indices = np.asarray(subset_test_source_indices)
        dump_embeddings(subset_test_source_indices, os.path.join(base_path, subset_test_source_idx_file))

    random.seed(params.model['seed'])
    random.shuffle(train)

    for _size in number_of_files_for_train:
        subset_train = train[:_size]
        subset_train_source_embeddings, subset_train_labels, subset_train_target_embeddings, subset_train_source_indices = zip(*subset_train)
        subset_train_source_embeddings_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]].format('train_recall',_size)
        subset_train_target_embeddings_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]].format('train_recall', _size)
        subset_train_source_label_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_LABELS"]].format('train_recall', _size)
        subset_train_source_idx_file = params.files['subset_file_format'][KN_FILE_NAMES["KN_SOURCE_IDX"]].format('train_recall',_size)

        subset_train_source_embeddings = np.asarray(subset_train_source_embeddings)
        dump_embeddings(subset_train_source_embeddings, os.path.join(base_path, subset_train_source_embeddings_file))
        subset_train_labels = np.asarray(subset_train_labels)
        dump_embeddings(subset_train_labels, os.path.join(base_path, subset_train_source_label_file), dtype="int32")
        subset_train_target_embeddings = np.asarray(subset_train_target_embeddings)
        dump_embeddings(subset_train_target_embeddings, os.path.join(base_path, subset_train_target_embeddings_file))
        subset_train_source_indices = np.asarray(subset_train_source_indices)
        dump_embeddings(subset_train_source_indices, os.path.join(base_path, subset_train_source_idx_file))

    # update params for new values
    params.files['test_subset_recall'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]] = subset_test_source_embeddings_file
    params.files['test_subset_recall'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]] = subset_test_target_embeddings_file
    params.files['test_subset_recall'][KN_FILE_NAMES["KN_SOURCE_LABELS"]] = subset_test_source_label_file
    params.files['test_subset_recall'][KN_FILE_NAMES["KN_SOURCE_IDX"]] = subset_test_source_idx_file

    params.files['train_subset_recall'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]] = subset_train_source_embeddings_file
    params.files['train_subset_recall'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]] = subset_train_target_embeddings_file
    params.files['train_subset_recall'][KN_FILE_NAMES["KN_SOURCE_LABELS"]] = subset_train_source_label_file
    params.files['train_subset_recall'][KN_FILE_NAMES["KN_SOURCE_IDX"]] = subset_train_source_idx_file

    params.files['test_subset_loss'][KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]] = subset_test_source_embeddings_file
    params.files['test_subset_loss'][KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]] = subset_test_target_embeddings_file

    params.files['splitter']['num_labels'] = num_labels
    params.files['splitter']['train_size'] = train_source_embeddings.shape[0]
    params.files['splitter']['test_size'] = test_source_embeddings.shape[0]

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
    analysis[counts_lower_than] = (df_counts_lower_than.shape[0], 100 * (df_counts_lower_than.shape[0] /
                                                                         df_with_count_of_labels.shape[0]))

    counts_between = 'counts_between_' + str(int(analysis['mean_of_count'])) + '_' + str(
        math.ceil(analysis['mean_of_count']))
    df_counts_between = df_with_count_of_labels[(df_with_count_of_labels['count'] >= int(analysis['mean_of_count'])) & (
                df_with_count_of_labels['count'] <= math.ceil(analysis['mean_of_count']))]

    analysis[counts_between] = (df_counts_between.shape[0], 100 * (df_counts_between.shape[0] /
                                                                   df_with_count_of_labels.shape[0]))

    counts_greater = 'counts_greater_than_' + str(math.ceil(analysis['mean_of_count']))
    df_counts_greater_than = df_with_count_of_labels[
        df_with_count_of_labels['count'] > math.ceil(analysis['mean_of_count'])]

    analysis[counts_greater] = (df_counts_greater_than.shape[0], 100 * (df_counts_greater_than.shape[0] /
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

    # ideal_number_of_paragraphs_for_each_question_for_debug
    analysis['K'] = int(analysis['mean_of_count']) if analysis[counts_equal_to_lower] > analysis[
        counts_equal_to_higher] else math.ceil(analysis['mean_of_count'])
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


def calculate_recalls(questions, paragraphs, labels, params, k=None, distance_type='cosine'):
    """
     question [n x d] tensor of n rows with d dimensions
     paragraphs [m x d] tensor of n rows with d dimensions
     params config
     returns:
     loss : scalar value
     """

    recalls = []  # tf.zeros([len(params.recall_top_k), 1], tf.float32)
    if distance_type == 'cosine':
        questions, labels, paragraphs, scores = pairwise_expanded_cosine_similarities(questions, labels, paragraphs)
        number_of_questions = tf.to_int64(tf.shape(questions)[1])
    else:
        scores= pairwise_euclidean_distances(questions, paragraphs)
        number_of_questions = tf.to_int64(tf.shape(questions)[0])
    for _k in [k] if k is not None else params.recall:
        with tf.name_scope('top_k_{}'.format(_k)) as k_scope:
            founds, _, __ = calculate_recall_top_k(scores, labels, _k, distance_type)
            total_founds_in_k = tf.reduce_sum(founds, name='{}_reduce_sum'.format(_k))
            recalls.append(total_founds_in_k)

    recalls = tf.stack(recalls)

    return recalls, (recalls / number_of_questions)

def pairwise_expanded_cosine_similarities(questions, labels, paragraphs):
    # in order to support batch_size feature, we expanded dims for 1
    paragraphs = tf.expand_dims(paragraphs, axis=0)
    labels = tf.expand_dims(labels, axis=0)
    questions = tf.expand_dims(questions, axis=0)
    # now question, paragraphs pairwise calculation
    cosine_similarities = pairwise_cosine_sim(questions, paragraphs)
    return questions, labels, paragraphs, cosine_similarities

def calculate_recall_top_k(scores, labels, k, distance_type = 'cosine'):


    if distance_type == 'cosine':
        axis = 2
    else:
        scores = tf.negative(scores)
        axis = 1

    # find the top_k paragraphs for each question
    top_k = tf.nn.top_k(scores, k=k, name='top_k_top_k_{}'.format(k))

    # is groundtruth label is in these top_k paragraph
    equals = tf.equal(top_k.indices, labels, name='equal_top_k_{}'.format(k))

    # cast the equals to int32 to count the non zero ones because if it is equal,
    # there is only one 1 for each question among paragraphs,
    # then label is in top k
    casted_equal = tf.cast(equals, dtype=tf.int32, name='casted_equal_top_k_{}'.format(k))
    founds = tf.squeeze(
        tf.count_nonzero(casted_equal, axis=axis, name='sq_top_k_{}'.format(k)))

    if distance_type == 'cosine':
        closest_labels = tf.squeeze(top_k.indices, axis=0)
        distances = tf.squeeze(top_k.values, axis=0)
        labels = tf.squeeze(labels, axis=0)
    else:
        closest_labels = top_k.indices
        distances = tf.negative(top_k.values)

    tmp_indices = tf.where(tf.equal(closest_labels, labels))
    found_orders = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
    return founds,closest_labels, distances

def next_batch(begin_indx, batch_size, questions, paragraphs, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    if begin_indx + batch_size <= questions.shape[0]:
        idx = np.arange(begin_indx, begin_indx + batch_size)
    else:
        idx = np.arange(begin_indx, questions.shape[0])
    # np.random.shuffle(idx)
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
    # questions = tf.constant(_q)
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
        else:  # type == 'q2p':
            k = 1
        top_k = tf.nn.top_k(dist, k=k, name='top_k_{}'.format(1))
        values = tf.reshape(tf.reduce_max(top_k.values, axis=2), shape=[tf.shape(sub_set_)[1], 1])
        if input_type == 'p2p':
            par_indices = tf.reshape(tf.squeeze(top_k.indices[:, :, 1]), shape=[-1, 1])
        else:  # type == 'q2p':
            par_indices = tf.reshape(tf.squeeze(top_k.indices), shape=[-1, 1])
        scores = tf.reshape(tf.squeeze(values), shape=[-1, 1])

    return scores, par_indices


def question_to_closest_distance(question_embeddings, paragraph_embeddings, batch_size, sess, closest_distance_op,
                                 question_tensor, paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]

        batch_distances, par_indices = sess.run(closest_distance_op, feed_dict={
            question_tensor: ques,
            paragraph_tensor: paragraph_embeddings,
        })
        # batch_distances[:,0] -> closest distances
        # batch_distances[:,1] -> indices
        distances = np.append(distances, batch_distances)
    distances = distances.reshape(-1, 1)
    return distances


def question_to_ground_truth_distance(question_embeddings, paragraph_embeddings, batch_size, sess,
                                      euclidean_distance_op, question_tensor,
                                      paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]
        pars = paragraph_embeddings[start:end]

        batch_distances = sess.run(euclidean_distance_op, feed_dict={
            question_tensor: ques,
            paragraph_tensor: pars,
        })
        distances = np.append(distances, batch_distances)
    distances = distances.reshape(-1, 1)
    return distances


def question_to_random_paragraph_distance(question_embeddings, paragraph_embeddings, batch_size, sess,
                                          euclidean_distance_op, question_tensor,
                                          paragraph_tensor):
    iter_size = math.ceil(question_embeddings.shape[0] / batch_size)
    #random.seed(params.eval_seed)
    np.random.shuffle(paragraph_embeddings)
    distances = np.array([])
    distances = distances.reshape(-1, 1)
    for _ in range(0, iter_size):
        start = _ * batch_size
        end = start + batch_size
        ques = question_embeddings[start:end]
        pars = paragraph_embeddings[start:end]
        #random.seed(params.eval_seed)
        np.random.shuffle(pars)
        batch_distances = sess.run(euclidean_distance_op, feed_dict={
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
    shape[shape == None] = -1
    shape[axis] = 1

    nd = len(shape)

    for i in indices:
        _slice = [slice(None)] * nd
        _slice[axis] = slice(i, i + 1)
        slices.append(tf.reshape(tensor[_slice], shape))

    return tf.concat(slices, axis=axis)


def get_variable_name_as_str(variable):
    return [k for k, v in locals().items() if v == variable][0]


def create_execution_name(params):
    model_name = params.model['active_model']
    model_params = params.model[model_name]
    num_epoch = "epoch_{}".format(params.model["num_epochs"])
    learning_rate = "lr_{}".format(params.optimizer["learning_rate"])
    margin = "mar_{}".format(params.loss['margin'])
    keep_prob = "keep"
    weight_decay = "wd"
    embedding_dim = "dim"
    scaling_factor = "sf"
    init_seed = "seed"
    loss = 'loss_v{}'.format(params.loss['version'])
    if type(model_params) is dict:
        model_params = [model_params]
    for indx, layers in enumerate(model_params):
        try:
            embedding_dim = embedding_dim + "_" + str(layers["fc_relu_embedding_dim"])
        except:
            embedding_dim = embedding_dim + "_" + str(layers["embedding_dim"])

        try:
            keep_prob = keep_prob + "_" + str(layers['keep_prob'])
        except:
            keep_prob = keep_prob + "_no"

        try:
            init_seed = init_seed + "_" + str(layers['initializer_seed'])
        except:
            init_seed = ""

        try:
            weight_decay = weight_decay + "_" + str(layers['weight_decay'])
        except:
            weight_decay = ""
        try:
            scaling_factor = scaling_factor + "_" + str(layers['scaling_factor'])
        except:
            scaling_factor = ""

    layers = "layers_{}".format(len(model_params))

    execution_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(model_name,
                                       layers,
                                       num_epoch,
                                       margin,
                                       scaling_factor,
                                       weight_decay,
                                          learning_rate,
                                          embedding_dim,
                                          keep_prob,
                                          init_seed,loss)

    if len(execution_name) > 250:
        execution_name = execution_name[0:251]
    return execution_name


def plot_tensorflow_log(path, metric_name):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies =   event_acc.Scalars(metric_name)
    validation_accuracies = event_acc.Scalars(metric_name)

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label=metric_name)
    plt.plot(x, y[:,1], label=metric_name)

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()

def save_as_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_as_shelve(obj, path):
    with shelve.open(path) as myShelve:
        myShelve.update(obj)

def load_from_shelve(path):
    obj = None
    with shelve.open(path) as myShelve:
        obj = myShelve
    return obj

def process_squad_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total,_i_para  = 0, 0
    questions = []
    paragraphs = []
    question_to_paragraph = []
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            title = article["title"]
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                paragraphs.append(context)
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    questions.append(ques)
                    question_to_paragraph.append(_i_para)
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, 'ques': ques,"answers": answer_texts, "uuid": qa["id"], 'title': title}
                _i_para += 1
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples, questions, paragraphs, question_to_paragraph

def load_module(module_url = "https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False):
    if trainable:
        embed = hub.Module(module_url, trainable)
    else:
        embed = hub.Module(module_url)
    return embed

def create_idf_matrix(tokenized_questions, tokenized_paragraphs, token2idfweight):
    idf_matrix = []
    for sentence in tokenized_questions + tokenized_paragraphs:
        for word in sentence:
            idf_matrix.append(token2idfweight[word])

    idf_matrix = np.asarray(idf_matrix)
    idf_matrix = idf_matrix.reshape(idf_matrix.shape[0], 1,1)
    return idf_matrix

def transform_to_idf_weigths(tokenized_questions, tokenized_paragraphs, tokenizer, questions_nontokenized,paragraphs_nontokenized):
    tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenizer)
    tfidf.fit(questions_nontokenized + paragraphs_nontokenized)
    max_idf = max(tfidf.idf_)
    token2idfweight = defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    idf_vec = create_idf_matrix(tokenized_questions, tokenized_paragraphs, token2idfweight)
    return token2idfweight, idf_vec

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def dump_tokenized_contexts(tokenized_contexts:list, file_path:str):
    with open(file_path, 'w') as fout:
        for context in tokenized_contexts:
            fout.write(' '.join(context) + '\n')

def tokenize_contexts(contexts:list):
    tokenized_context = [word_tokenize(context.strip()) for context in contexts]
    return tokenized_context

def prepare_squad_objects(squad_file,dataset_type, is_dump_during_execution=False,
                          paragraphs_file=None, questions_file=None, mapping_file=None):
    print(100 * '*')
    print('Parsing Started')
    start = datetime.datetime.now()

    word_counter, char_counter = Counter(), Counter()
    examples, eval, questions, paragraphs, q_to_ps = process_squad_file(squad_file,
                                                                        dataset_type,
                                                                        word_counter,
                                                                        char_counter)

    print('# of Paragraphs in {} : {}'.format(dataset_type, len(paragraphs)))
    print('# of Questions in {} : {}'.format(dataset_type, len(questions)))
    print('# of Q_to_P {} : {}'.format(dataset_type, len(q_to_ps)))

    print(20 * '-')
    print('Paragraphs: Tokenization and Saving Tokenization Started in {}'.format(dataset_type))
    tokenized_paragraphs = tokenize_contexts(paragraphs)
    print('# of Tokenized Paragraphs in {} : {}'.format(dataset_type, len(tokenized_paragraphs)))
    print(20 * '-')
    print('Questions: Tokenization and Saving Tokenization Started in {}'.format(dataset_type))
    tokenized_questions = tokenize_contexts(questions)
    print('# of Tokenized Questions in {} : {}'.format(dataset_type, len(tokenized_questions)))

    if is_dump_during_execution:
        dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file)
        dump_tokenized_contexts(tokenized_questions, questions_file)
        dump_mapping_data(q_to_ps, mapping_file)
    end = datetime.datetime.now()
    print('Parsing Ended in {} minutes'.format((end - start).seconds / 60))
    print(100 * '*')

    tokenized_questions, tokenized_paragraphs = fixing_the_token_problem(tokenized_questions, tokenized_paragraphs)
    paragraphs_nontokenized = [" ".join(context) for context in tokenized_paragraphs]
    questions_nontokenized = [" ".join(context) for context in tokenized_questions]

    return tokenized_questions, tokenized_paragraphs, questions_nontokenized, paragraphs_nontokenized



def fixing_the_token_problem(tokenized_questions, tokenized_paragraphs):
    # fixing the '' problem:
    fixed_tokenized_question = []
    for indx, question in enumerate(tokenized_questions):
        tokens = []
        for token in question:
            t = token.strip()
            if t != "":
                tokens.append(t)
        fixed_tokenized_question.append(tokens)

    fixed_tokenized_paragraph = []
    for indx, paragraph in enumerate(tokenized_paragraphs):
        tokens = []
        for token in paragraph:
            t = token.strip()
            if t != "":
                tokens.append(t)
        fixed_tokenized_paragraph.append(tokens)
    return fixed_tokenized_question, fixed_tokenized_paragraph
def generate_document_embedding_guideline(tokenized_questions, tokenized_paragraphs, is_dump = False,
                                          token_embeddings_guideline_file = None, tokens_ordered_file = None):
    print(100 * '*')
    print('Index of tokens in each document is getting identified....')
    start = datetime.datetime.now()
    document_embedding_guideline = defaultdict()
    corpus_as_tokens = []

    for i, sentence in enumerate(tokenized_questions + tokenized_paragraphs):
        document_embedding_guideline[i] = defaultdict()
        document_embedding_guideline[i]['start_index'] = len(corpus_as_tokens)
        document_embedding_guideline[i]['end_index'] = len(corpus_as_tokens) + len(sentence)
        if i >= len(tokenized_questions):
            document_embedding_guideline[i]['type'] = 'p'
        else:
            document_embedding_guideline[i]['type'] = 'q'
        for token in sentence:
            corpus_as_tokens.append(token)
    if is_dump:
        save_as_pickle(document_embedding_guideline, token_embeddings_guideline_file)
        save_as_pickle(corpus_as_tokens, tokens_ordered_file)
    # del document_embedding_guideline
    # del corpus_as_tokens
    print("Total tokens in the corpus: {}".format(len(corpus_as_tokens)))
    end = datetime.datetime.now()
    print('Index of tokens in each document is getting saved in {} minutes'.format((end - start).seconds / 60))
    print(100 * '*')
    return document_embedding_guideline, corpus_as_tokens
def token_to_document_embeddings(tokenized_questions, tokenized_paragraphs,token_embeddings, token_embeddings_guideline, weight_matrix=None):
    questions_embeddings = []
    paragraphs_embeddings = []
    documents = tokenized_questions + tokenized_paragraphs
    for _ in tqdm(range(len(documents))):
        str_index = token_embeddings_guideline[_]['start_index']
        end_index = token_embeddings_guideline[_]['end_index']
        d_type = token_embeddings_guideline[_]['type']
        if d_type == 'q':
            if weight_matrix is None:
                questions_embeddings.append(np.mean(token_embeddings[str_index:end_index, :], axis=0))
            else:
                questions_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_question_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))
        else:
            if weight_matrix is None:
                paragraphs_embeddings.append(np.mean(token_embeddings[str_index:end_index, :], axis=0))
            else:
                paragraphs_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_paragraph_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))

    del token_embeddings
    questions_embeddings = np.asarray(questions_embeddings)
    print('Question Embbeddings Shape 1:{}'.format(questions_embeddings.shape))
    paragraphs_embeddings = np.asarray(paragraphs_embeddings)
    print('Paragraph Embbeddings Shape 1:{}'.format(paragraphs_embeddings.shape))

    if weight_matrix is not None:
        questions_embeddings = np.multiply(questions_embeddings, weight_matrix)
        print('Question Embbeddings Shape 2:{}'.format(questions_embeddings.shape))
        paragraphs_embeddings = np.multiply(paragraphs_embeddings, weight_matrix)
        print('Paragraph Embbeddings Shape 2:{}'.format(paragraphs_embeddings.shape))

        # questions_embeddings = np.mean(questions_embeddings, axis=1)
        # print('Question Embbeddings Shape 3:{}'.format(questions_embeddings.shape))
        # paragraphs_embeddings = np.mean(paragraphs_embeddings, axis=1)
        # print('Paragraph Embbeddings Shape 3:{}'.format(paragraphs_embeddings.shape))

    return questions_embeddings, paragraphs_embeddings

def generate_and_dump_elmo_embeddings(documents,
                                      non_context,
                                     vocab_file_path,
                                     dataset_file_path,
                                     options_file_path,
                                     weights_file_path,
                                     embedding_file_path,
                                     size_of_each_partition
                                      ):
    '''
    ELMo usage example to write biLM embeddings for an entire dataset to
    a file.
    '''
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bilm import dump_usage_cache_embeddings

    # convert documents into this format
    if non_context:
        new_documents = [[doc] for doc in documents]
    else:
        new_documents = documents

    # Create the dataset file.
    with open(dataset_file_path, 'w') as fout:
        for sentence in new_documents:
            cleaned_sentence = [word.replace('\n', ' ') for word in sentence]
            fout.write(' '.join(cleaned_sentence) + '\n')

    if non_context:
        # Dump the embeddings to a file. Run this once for your dataset.
        embeddings = dump_usage_cache_embeddings(
            vocab_file_path,
            dataset_file_path,
            options_file_path,
            weights_file_path,
            embedding_file_path,
            size_of_each_partition
        )
    else:
        dump_usage_cache_embeddings(
            vocab_file_path,
            dataset_file_path,
            options_file_path,
            weights_file_path,
            embedding_file_path
        )


    #embeddings = load_embeddings(embedding_file_path)
    return embeddings


def generate_and_dump_contextualized_elmo_embeddings(tokenized_questions,
                                                     tokenized_paragraphs,
                                                      vocab_file_path,
                                                      options_file_path,
                                                      weights_file_path,
                                                      token_embedding_file_path,
                                                      embedding_file_path
                                      ):
    '''
    ELMo usage example to write biLM embeddings for an entire dataset to
    a file.
    '''
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

    dump_token_embeddings(
        vocab_file_path, options_file_path, weights_file_path, token_embedding_file_path
    )

    tf.reset_default_graph()

    ## Now we can do inference.
    # Create a TokenBatcher to map text to token ids.
    batcher = TokenBatcher(vocab_file_path)

    # Input placeholders to the biLM.
    paragraph_token_ids = tf.placeholder('int32', shape=(None, None))
    question_token_ids = tf.placeholder('int32', shape=(None, None))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(
        options_file_path,
        weights_file_path,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file_path
    )

    paragraph_embeddings_op = bilm(paragraph_token_ids)
    question_embeddings_op = bilm(question_token_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    # Our SQuAD model includes ELMo at both the input and output layers
    # of the task GRU, so we need 4x ELMo representations for the question
    # and context at each of the input and output.
    # We use the same ELMo weights for both the question and context
    # at each of the input and output.
    # elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

    elmo_paragraph_input = weight_layers(
        'input', paragraph_embeddings_op, l2_coef=0.0
    )
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_question_input = weight_layers(
            'input', question_embeddings_op, l2_coef=0.0
        )

    elmo_paragraph_output = weight_layers(
        'output', paragraph_embeddings_op, l2_coef=0.0
    )
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_question_output = weight_layers(
            'output', question_embeddings_op, l2_coef=0.0
        )


    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        for doc_id, paragraph in enumerate(tokenized_paragraphs):
            p = [paragraph]
            paragraph_ids = batcher.batch_sentences(p)

            elmo_paragraph_input_, = sess.run(
                [elmo_paragraph_input['weighted_op']],
                feed_dict={paragraph_token_ids: paragraph_ids}
            )
            new_embedding = np.swapaxes(elmo_paragraph_input_, 0, 1)
            with h5py.File(
                    embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                             '//ELMO_CONTEXT_OLD_API_ELMO_INPUT_EMBEDDINGS//parapgraphs//'),
                    'w') as fout:
                ds = fout.create_dataset(
                    'embeddings',
                    new_embedding.shape, dtype='float32',
                    data=new_embedding
                )
            print('{} is dumped'.format(
                    embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                             '//ELMO_CONTEXT_OLD_API_ELMO_INPUT_EMBEDDINGS//paragraphs//')))

            elmo_paragraph_output_ = sess.run(
                [elmo_paragraph_output['weighted_op']],
                feed_dict={paragraph_token_ids: paragraph_ids}
            )

            new_embedding = np.swapaxes(elmo_paragraph_output_, 0, 1)
            with h5py.File(
                    embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                    '//ELMO_CONTEXT_OLD_API_ELMO_OUTPUT_EMBEDDINGS//paragraphs//'),
                    'w') as fout:
                ds = fout.create_dataset(
                    'embeddings',
                    new_embedding.shape, dtype='float32',
                    data=new_embedding
                )
            print('{} is dumped'.format(
                embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                '//ELMO_CONTEXT_OLD_API_ELMO_OUTPUT_EMBEDDINGS//paragraphs//')))

        # Create batches of data.
        for question in tokenized_questions:
            q = [question]
            question_ids = batcher.batch_sentences(q)

            elmo_question_input_ = sess.run(
                [elmo_question_input['weighted_op']],
                feed_dict={question_token_ids: question_ids}
            )

            #embeddings = elmo_question_input_[0, :, :]
            new_embedding = np.swapaxes(elmo_question_input_, 0, 1)
            with h5py.File(
                    embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                    '//ELMO_CONTEXT_OLD_API_ELMO_INPUT_EMBEDDINGS//questions//'),
                    'w') as fout:
                ds = fout.create_dataset(
                    'embeddings',
                    new_embedding.shape, dtype='float32',
                    data=new_embedding
                )
            print('{} is dumped'.format(
                embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                '//ELMO_CONTEXT_OLD_API_ELMO_INPUT_EMBEDDINGS//questions//')))

            elmo_question_output_ = sess.run(
                [elmo_question_output['weighted_op']],
                feed_dict={question_token_ids: question_ids}
            )

            # embeddings = elmo_question_output_[0, :, :]
            new_embedding = np.swapaxes(elmo_question_output_, 0, 1)
            with h5py.File(
                    embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                    '//ELMO_CONTEXT_OLD_API_ELMO_OUTPUT_EMBEDDINGS//questions//'),
                    'w') as fout:
                ds = fout.create_dataset(
                    'embeddings',
                    new_embedding.shape, dtype='float32',
                    data=new_embedding
                )
            print('{} is dumped'.format(
                embedding_file_path.replace('@@', 'doc_' + str(doc_id)).replace('/./',
                                                                                '//ELMO_CONTEXT_OLD_API_ELMO_OUTPUT_EMBEDDINGS//questions//')))

        # Compute ELMo representations (here for the input only, for simplicity).










    # embeddings = load_embeddings(embedding_file_path)
    #return embeddings