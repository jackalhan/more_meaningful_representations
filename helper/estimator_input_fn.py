"""Create the input data pipeline using `tf.data`"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.estimator_dataset as ds
import helper.utils as UTIL
import tensorflow as tf
import numpy as np


def parser(questions, document_length, baseline_question_embeddings, paragraph_as_embeddings, paragraph_as_label):
    features = {"questions": questions,
                "document_length": document_length,
                "baseline_question_embeddings": baseline_question_embeddings}
    labels = {"paragraph_as_embeddings": paragraph_as_embeddings,
              "paragraph_as_label": paragraph_as_label}
    return features, labels


def prepare_data(base_data_path, params, mode):
    """
       START: DATA PREPARATION
       """

    # TRAIN

    train_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
                                                             params.files['train_loss']['question_labels']))
    train_question_label_indx = train_question_label_indx.astype(int)

    train_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
                                                       params.files['train_loss']['question_embeddings']))
    train_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
                                                         params.files['train_loss']['paragraph_embeddings']))

    # TEST RECALL

    test_recall_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
                                                             params.files['test_subset_recall']['question_labels']))
    test_recall_question_label_indx = test_recall_question_label_indx.astype(int)

    test_recall_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
                                                       params.files['test_subset_recall']['question_embeddings']))

    test_recall_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
                                                         params.files['test_subset_recall']['paragraph_embeddings']))

    # TRAIN RECALL

    train_recall_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
                                                                    params.files['train_subset_recall'][
                                                                        'question_labels']))
    train_recall_question_label_indx = train_recall_question_label_indx.astype(int)

    train_recall_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
                                                              params.files['train_subset_recall'][
                                                                  'question_embeddings']))

    train_recall_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
                                                                params.files['train_subset_recall'][
                                                                    'paragraph_embeddings']))

    # ----------------------------------------------------------------

    if params.model['model_type'].lower() == 'conv':
        x_train_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
                                                params.files['train_loss']['question_x_train']))

        y_train_paragraph_as_embeddings= train_question_labels
        y_train_paragraph_as_label = train_question_label_indx

        del train_question_labels, train_question_label_indx
        print("x_train_questions shape is {}".format(x_train_questions.shape))

        x_test_recall_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
                                                params.files['test_subset_recall']['question_x_valid']))
        y_test_recall_paragraph_as_embeddings = test_recall_question_labels
        y_test_recall_paragraph_as_label = test_recall_question_label_indx
        del test_recall_question_labels, test_recall_question_label_indx
        print("x_test_recall_questions shape is {}".format(x_test_recall_questions.shape))

        x_train_recall_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
                                                       params.files['train_subset_recall']['question_x_train_valid']))
        y_train_recall_paragraph_as_embeddings = train_recall_question_labels
        y_train_recall_paragraph_as_label = train_recall_question_label_indx
        del train_recall_question_labels, train_recall_question_label_indx
        print("x_train_recall_questions shape is {}".format(x_train_recall_questions.shape))

        voc_to_indx = UTIL.load_from_pickle(os.path.join(base_data_path,
                                                    params.files['voc_to_indx']))
        vocab_size = len(voc_to_indx)
        print('Total words: %d' % vocab_size)
        params.files['questions_vocab_size'] = vocab_size

        max_document_len = params.files['max_document_len']

        if params.files['word_embeddings'] is None:
            params.model['conv_embedding_initializer'] = tf.truncated_normal_initializer(
                seed=params.model['initializer_seed'],
                stddev=0.1)
        else:
            word_embeddings = UTIL.load_word_embeddings(os.path.join(base_data_path, params.files['word_embeddings']),
                                                   voc_to_indx,
                                                   params.files['pre_trained_files']['embedding_dim'])

            def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
                assert dtype is tf.float32
                return word_embeddings

            params.model['conv_embedding_initializer'] = my_initializer
        """
        END: DATA PREPARATION
        """

        x_train_document_length = np.array([min(len(x), max_document_len) for x in x_train_questions])
        x_test_recall_document_length = np.array([min(len(x), max_document_len) for x in x_test_recall_questions])
        x_train_recall_document_length= np.array([min(len(x), max_document_len) for x in x_train_recall_questions])

        x_train_baseline_question_embeddings = train_org_questions
        x_test_recall_baseline_question_embeddings = test_recall_org_questions
        x_train_recall_baseline_question_embeddings = train_recall_org_questions
        del train_org_questions, test_recall_org_questions, train_recall_org_questions

    else:
        x_train_questions = train_org_questions
        y_train_paragraph_as_embeddings = train_question_labels
        y_train_paragraph_as_label = train_question_label_indx

        del train_question_labels, train_question_label_indx
        print("x_train_questions shape is {}".format(x_train_questions.shape))

        x_test_recall_questions =test_recall_org_questions
        y_test_recall_paragraph_as_embeddings = test_recall_question_labels
        y_test_recall_paragraph_as_label = test_recall_question_label_indx
        del test_recall_question_labels, test_recall_question_label_indx
        print("x_test_recall_questions shape is {}".format(x_test_recall_questions.shape))

        x_train_recall_questions = train_recall_org_questions
        y_train_recall_paragraph_as_embeddings = train_recall_question_labels
        y_train_recall_paragraph_as_label = train_recall_question_label_indx
        del train_recall_question_labels, train_recall_question_label_indx
        print("x_train_recall_questions shape is {}".format(x_train_recall_questions.shape))
        del train_org_questions, test_recall_org_questions, train_recall_org_questions

        x_train_document_length = None
        x_test_recall_document_length = None
        x_train_recall_document_length = None

        x_train_baseline_question_embeddings = train_org_questions
        x_test_recall_baseline_question_embeddings = test_recall_org_questions
        x_train_recall_baseline_question_embeddings = train_recall_org_questions

    if mode == 'train':
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_questions,
             x_train_document_length,
             x_train_baseline_question_embeddings,
             y_train_paragraph_as_embeddings,
             y_train_paragraph_as_label))
        if params.model["shuffle"]:
            dataset = dataset.shuffle(buffer_size=x_train_questions.shape[0])
        dataset = dataset.batch(params.model["batch_size"])
        dataset = dataset.map(parser)
        # dataset = dataset.repeat()
        # dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    elif mode == 'test_recall':
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_test_recall_questions,
             x_test_recall_document_length,
             x_test_recall_baseline_question_embeddings,
             y_test_recall_paragraph_as_embeddings,
             y_test_recall_paragraph_as_label))
        dataset = dataset.batch(params.files["splitter"]["test_subset_size"])
        dataset = dataset.map(parser)
        # dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    elif mode == 'train_recall':
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_recall_questions,
             x_train_recall_document_length,
             x_train_recall_baseline_question_embeddings,
             y_train_recall_paragraph_as_embeddings,
             y_train_recall_paragraph_as_label))
        dataset = dataset.batch(params.files["splitter"]["train_subset_size"])
        dataset = dataset.map(parser)
        # dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    else:
        prediction_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
                                                                       params.files['prediction'][
                                                                           'question_embeddings']))

        dataset = tf.data.Dataset.from_tensor_slices(
            (prediction_org_questions,
             None,
             None,
             None,
             None))
        dataset = dataset.batch(params.model["batch_size"])
        dataset = dataset.map(parser)
        # dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def train_input_fn(base_data_path, params):
    return prepare_data(base_data_path, params, 'train')


def test_recall_input_fn(base_data_path, params):
    return prepare_data(base_data_path, params, 'test_recall')

def train_recall_input_fn(base_data_path, params):
    return prepare_data(base_data_path, params, 'train_recall')

def predict_input_fn(base_data_path, params):
    return prepare_data(base_data_path, params, 'predict')



# def train_input_fn(base_data_path, params):
#     """Train input function for the dataset.
#
#     Args:
#         base_data_path: (string) base path for all data
#         params: (Params) contains all the details of the execution including names of the data
#     """
#
#     dataset = ds.get_dataset(os.path.join(base_data_path, params.files['train_loss']['question_embeddings']),
#                              os.path.join(base_data_path, params.files['train_loss']['question_labels']),
#                              os.path.join(base_data_path, params.files['train_loss']['paragraph_embeddings']),
#                              params.files['pre_trained_files']['embedding_dim'],
#                              including_target=True)
#     if params.model["shuffle"]:
#         dataset = dataset.shuffle(params.files['splitter']["train_size"])  # whole dataset into the buffer
#     #dataset = dataset.repeat(1)  # repeat for multiple epochs
#     dataset = dataset.batch(params.model["batch_size"])
#     #dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
#
# def test_input_fn(base_data_path, params):
#     """Test input function for the dataset.
#
#     Args:
#         base_data_path: (string) base path for all data
#         params: (Params) contains all the details of the execution including names of the data
#     """
#
#     dataset = ds.get_dataset(os.path.join(base_data_path, params.files['test_subset_loss']['question_embeddings']),
#                              None,
#                              os.path.join(base_data_path, params.files['test_subset_loss']['paragraph_embeddings']),
#                              params.files['pre_trained_files']['embedding_dim'],
#                              including_target=True)
#     dataset = dataset.batch(params.files["splitter"]["test_subset_size"])
#     #dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
#
# def live_input_fn(base_data_path, params):
#     """Live input function for the dataset.
#
#     Args:
#         base_data_path: (string) base path for all data
#         params: (Params) contains all the details of the execution including names of the data
#     """
#     dataset = ds.get_dataset(os.path.join(base_data_path, params.files['prediction']['question_embeddings']),
#                              None,
#                              None,
#                              params.files['pre_trained_files']['embedding_dim'],
#                              including_target=False)
#
#     dataset = dataset.batch(params.model["batch_size"])
#     dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
#     return dataset

