"""Create the input data pipeline using `tf.data`"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.estimator_dataset as ds
import helper.utils as UTIL
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing import sequence

class DataBuilder():
    """ DataBuilder class for all type of estimator input_fn
    """

    def __init__(self, base_path, params, verbose):
        self.params = params
        self.base_path = base_path
        self.verbose = verbose
        self._init_data()
    def _parser_conv(self_, source_embeddings,
               source_document_length,
               baseline_source_embeddings,
               target_embeddings,
               target_labels):

        features = {"source_embeddings": source_embeddings,
                    "source_document_length": source_document_length,
                    "baseline_source_embeddings": baseline_source_embeddings}
        labels = {"target_embeddings": target_embeddings,
                  "target_labels": target_labels}
        return features, labels

    def _parser_non_conv(self_, source_embeddings,
               target_embeddings,
               target_labels):

        features = {"source_embeddings": source_embeddings}
        labels = {"target_embeddings": target_embeddings,
                  "target_labels": target_labels}
        return features, labels

    def _parser_estimate_conv(self, source_embeddings,source_document_length):
        features = {"source_embeddings": source_embeddings,
                    "source_document_length": source_document_length}
        return features

    def _parser_estimate_non_conv(self, source_embeddings):
        features = {"source_embeddings": source_embeddings}
        return features

    def _init_data(self):
        """
        :param verbose: list ALL_DATA or one or combination of the followings {TRAIN, TRAIN_RECALL, TEST_RECALL, PREDICT}
        :return:
        """
        self.KN_FILE_NAMES = UTIL.get_file_key_names_for_execution(self.params)
        self.data_dir = os.path.join(self.base_path, self.KN_FILE_NAMES['DIR'])
        print("Data is getting loaded for the {}".format(self.KN_FILE_NAMES['DIR']))
        verbose = [item.lower() for item in self.verbose]
        # KN_FILE_NAMES["KN_SOURCE_EMBEDDINGS"]
        # KN_FILE_NAMES["KN_SOURCE_LABELS"]
        # KN_FILE_NAMES["KN_SOURCE_IDX"]
        # KN_FILE_NAMES["KN_TARGET_EMBEDDINGS"]
        self._get_tokenized_data()
        if self.params.model['model_type'].lower() == 'conv':
            self._build_vocabulary()
            self._embeddings_initializer()

        if 'all_data' in verbose:
            self._load_train_data()
            self._load_train_recall_data()
            self._load_test_recall_data()
            self._load_predict_data()
        else:
            if 'train' in verbose:
                self._load_train_data()
            if 'train_recall' in verbose:
                self._load_train_recall_data()
            if 'test_recall' in verbose:
                self._load_test_recall_data()
            if 'predict' in verbose:
                self._load_predict_data()

    def _embeddings_initializer(self):
        if self.params.files['word_embeddings'] is None:
            self.params.model['conv_embedding_initializer'] = tf.truncated_normal_initializer(
                seed=self.params.model['initializer_seed'],
                stddev=0.1)
        else:
            word_embeddings = UTIL.load_word_embeddings(os.path.join(self.base_path,
                                                                     self.params.files['word_embeddings']),
                                                   self.voc_to_indx,
                                                   self.params.files['pre_trained_files']['embedding_dim'])

            def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
                assert dtype is tf.float32
                return word_embeddings

            self.params.model['conv_embedding_initializer'] = my_initializer
    def _get_tokenized_data(self):

        dev_tokenized_questions, \
        dev_tokenized_paragraphs, \
        dev_questions_nontokenized, \
        dev_paragraphs_nontokenized = UTIL.prepare_squad_objects(os.path.join(self.base_path, self.params.files['squad']['dev']), 'DEV')

        print('Len of Dev Questions: {}, Len of Dev Paragraphs: {}'.format(len(dev_tokenized_questions),
                                                                           len(dev_tokenized_paragraphs)))
        tokenized_questions = dev_tokenized_questions
        tokenized_paragraphs = dev_tokenized_paragraphs
        questions_nontokenized = dev_questions_nontokenized
        paragraphs_nontokenized = dev_paragraphs_nontokenized
        if 'dev' not in self.params.executor['data_dir'].lower():
            train_tokenized_questions, \
            train_tokenized_paragraphs, \
            train_questions_nontokenized, \
            train_paragraphs_nontokenized = UTIL.prepare_squad_objects(os.path.join(self.base_path, self.params.files['squad']['train']), 'TRAIN')
            print('Len of Train Questions: {}, Len of Train Paragraphs: {}'.format(len(train_tokenized_questions),
                                                                                   len(train_tokenized_paragraphs)))

            tokenized_questions = tokenized_questions + train_tokenized_questions
            tokenized_paragraphs = tokenized_paragraphs + train_tokenized_paragraphs
            questions_nontokenized = questions_nontokenized + train_questions_nontokenized
            paragraphs_nontokenized = paragraphs_nontokenized + train_paragraphs_nontokenized

        print('Len of Total Questions: {}, Len of Total Paragraphs: {}'.format(len(tokenized_questions),
                                                                               len(tokenized_paragraphs)))

        self._tokenized_questions = tokenized_questions
        self._tokenized_paragraphs = tokenized_paragraphs
        self._questions_nontokenized = questions_nontokenized
        self._paragraphs_nontokenized = paragraphs_nontokenized

    def _build_vocabulary(self):
        if self.KN_FILE_NAMES['dir'].lower().startswith('qu'):
            tokenized_sources = self._tokenized_questions
        else:
            tokenized_sources = self._tokenized_paragraphs
        indx_to_voc, self.voc_to_indx = UTIL.vocabulary_processor(tokenized_sources)
        self.vocab_size = len(self.voc_to_indx)
        print('Vocab Size: %d' % self.vocab_size)
        # params.files['questions_vocab_size'] = vocab_size
        print('vocabulary is build on {}'.format(self.KN_FILE_NAMES['dir']))

    def _obtain_tokenized_documents(self, source_indx):
        documents = []
        for indx in tqdm(source_indx):
            if self.KN_FILE_NAMES['dir'].lower().startswith('qu'):
                document = self._questions_nontokenized[indx]
            else:
                document = self._paragraphs_nontokenized[indx]
            documents.append(document)
        tokenized_documents = [document.split(' ') for document in documents]
        return tokenized_documents

    def _pad_documents(self, tokenized_documents):
        fitted_tokenized_documents= UTIL.fit_vocab_to_documents(tokenized_documents, self.voc_to_indx)
        padded_documents = sequence.pad_sequences(fitted_tokenized_documents,
                               maxlen=self.params.files['max_document_len'],
                               truncating='post',
                               padding='post',
                               value=0)
        padded_documents_lengths = np.array([min(len(x), self.params.files['max_document_len']) for x in padded_documents])
        return padded_documents, padded_documents_lengths

    def train_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            dataset = tf.data.Dataset.from_tensor_slices((self._train_source_embeddings,
                                                         self._train_source_embeddings_lengths,
                                                         self._train_baseline_source_embeddings,
                                                         self._train_target_embeddings,
                                                         self._train_source_labels))
            dataset = dataset.map(self._parser_conv)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self._train_source_embeddings,
                                                          self._train_target_embeddings,
                                                          self._train_source_labels))
            dataset = dataset.map(self._parser_non_conv)

        if self.params.model["shuffle"]:
            dataset = dataset.shuffle(buffer_size=self._train_source_embeddings.shape[0])
        dataset = dataset.batch(self.params.model["batch_size"])
        # dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=self.params.model["batch_size"], seed=1)
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        return dataset#iterator.get_next()

    def _load_train_data(self):
        self._train_source_labels, self._train_source_indx, self._train_source_embeddings, self._train_target_embeddings, self._train_all_target_embeddings = self._load_data('train_loss')
        self._train_baseline_source_embeddings = self._train_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._train_source_indx)
            self._train_source_embeddings, self._train_source_embeddings_lengths = self._pad_documents(tokenized_documents)
        else:
            self._train_source_embeddings_lengths = np.zeros([self._train_source_embeddings.shape[0], self._train_source_embeddings.shape[1]])

    def train_recall_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._train_recall_source_embeddings,
                 self._train_recall_source_embeddings_lengths,
                 self._train_recall_baseline_source_embeddings,
                 self._train_recall_target_embeddings,
                 self._train_recall_source_labels))
            dataset = dataset.map(self._parser_conv)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._train_recall_source_embeddings,
                 self._train_recall_target_embeddings,
                 self._train_recall_source_labels))
            dataset = dataset.map(self._parser_non_conv)

        dataset = dataset.batch(self.params.files["splitter"]["train_subset_size"])
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        return dataset #iterator.get_next()

    def _load_train_recall_data(self):
        self._train_recall_source_labels, self._train_recall_source_indx, self._train_recall_source_embeddings, self._train_recall_target_embeddings, self._train_recall_all_target_embeddings = self._load_data('train_subset_recall')
        self._train_recall_baseline_source_embeddings = self._train_recall_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._train_recall_source_indx)
            self._train_recall_source_embeddings, self._train_recall_source_embeddings_lengths = self._pad_documents(
                tokenized_documents)
        else:
            self._train_recall_source_embeddings_lengths = np.zeros(
                [self._train_recall_source_embeddings.shape[0], self._train_recall_source_embeddings.shape[1]])

    def test_recall_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._test_recall_source_embeddings,
                 self._test_recall_source_embeddings_lengths,
                 self._test_recall_baseline_source_embeddings,
                 self._test_recall_target_embeddings,
                 self._test_recall_source_labels))
            dataset = dataset.map(self._parser_conv)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._test_recall_source_embeddings,
                 self._test_recall_target_embeddings,
                 self._test_recall_source_labels))
            dataset = dataset.map(self._parser_non_conv)

        dataset = dataset.batch(self.params.files["splitter"]["test_subset_size"])

        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        return dataset #iterator.get_next()
    def _load_test_recall_data(self):
        self._test_recall_source_labels, self._test_recall_source_indx, self._test_recall_source_embeddings, self._test_recall_target_embeddings, self._test_recall_all_target_embeddings = self._load_data('test_subset_recall')
        self._test_recall_baseline_source_embeddings = self._test_recall_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._test_recall_source_indx)
            self._test_recall_source_embeddings, self._test_recall_source_embeddings_lengths = self._pad_documents(tokenized_documents)
        else:
            self._test_recall_source_embeddings_lengths = np.zeros(
                [self._test_recall_source_embeddings.shape[0], self._test_recall_source_embeddings.shape[1]])

    def predict_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._source_embeddings,
                 self._source_embeddings_lengths))
            dataset = dataset.map(self._parser_estimate_conv)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self._source_embeddings))
            dataset = dataset.map(self._parser_estimate_non_conv)

        dataset = dataset.batch(self.params.model["batch_size"])
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        return dataset #iterator.get_next()
    def _load_predict_data(self):
        self._source_embeddings = UTIL.load_embeddings(
            os.path.join(self.base_path, self.params.files['prediction']['source_embeddings']))
        if self.KN_FILE_NAMES['DIR'].lower().startswith('qu'):
            tokenized_documents = self._tokenized_questions
        else:
            tokenized_documents = self._tokenized_paragraphs
        if self.params.model['model_type'].lower() == 'conv':
            self._source_embeddings, self._source_embeddings_lengths = self._pad_documents(
                tokenized_documents)
        else:
            self._source_embeddings_lengths = np.zeros(
                [self._source_embeddings.shape[0], self._source_embeddings.shape[1]])

    def _load_data(self, data_type):
        source_labels = UTIL.load_embeddings(os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_LABELS']])).astype(int)
        source_indx = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_IDX']])).astype(int)
        source_embeddings = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_EMBEDDINGS']]))
        target_embeddings = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']]))
        try:
            all_target_embeddings = UTIL.load_embeddings(
                os.path.join(self.data_dir, self.params.files[data_type]["all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']]))
        except:
            print('{} is not found for {} type'.format("all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS'], data_type))
            all_target_embeddings = None
        return source_labels, source_indx, source_embeddings, target_embeddings,all_target_embeddings



# def train_input_fn(base_data_path, params):
#     # TRAIN
#
#     train_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                   params.files['train_loss']['question_labels']))
#     train_question_label_indx = train_question_label_indx.astype(int)
#
#     train_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                             params.files['train_loss']['question_embeddings']))
#     train_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                               params.files['train_loss']['paragraph_embeddings']))
#
#     if params.model['model_type'].lower() == 'conv':
#         max_document_len = params.files['max_document_len']
#         x_train_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
#                                                 params.files['train_loss']['question_x_train']))
#
#         y_train_paragraph_as_embeddings= train_question_labels
#         y_train_paragraph_as_label = train_question_label_indx
#         x_train_document_length = np.array([min(len(x), max_document_len) for x in x_train_questions])
#     else:
#         x_train_questions = train_org_questions
#         y_train_paragraph_as_embeddings = train_question_labels
#         y_train_paragraph_as_label = train_question_label_indx
#         x_train_document_length = tf.zeros([train_org_questions.shape[0], train_org_questions.shape[1]])
#     del train_question_labels, train_question_label_indx
#     print("x_train_questions shape is {}".format(x_train_questions.shape))
#     x_train_baseline_question_embeddings = train_org_questions
#     dataset = tf.data.Dataset.from_tensor_slices( (x_train_questions,
#                                                  x_train_document_length,
#                                                  x_train_baseline_question_embeddings,
#                                                  y_train_paragraph_as_embeddings,
#                                                  y_train_paragraph_as_label))
#     if params.model["shuffle"]:
#         dataset = dataset.shuffle(buffer_size=x_train_questions.shape[0])
#     dataset = dataset.batch(params.model["batch_size"])
#     dataset = dataset.map(parser_train)
#     # dataset = dataset.repeat()
#     # dataset = dataset.prefetch(1)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
#
# def test_recall_input_fn(base_data_path, params):
#     # TEST RECALL
#
#     test_recall_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                         params.files['test_subset_recall'][
#                                                                             'question_labels']))
#     test_recall_question_label_indx = test_recall_question_label_indx.astype(int)
#
#     test_recall_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                   params.files['test_subset_recall'][
#                                                                       'question_embeddings']))
#
#     test_recall_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                     params.files['test_subset_recall'][
#                                                                         'paragraph_embeddings']))
#
#
#     if params.model['model_type'].lower() == 'conv':
#         max_document_len = params.files['max_document_len']
#         x_test_recall_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
#                                                                      params.files['test_subset_recall'][
#                                                                          'question_x_valid']))
#         y_test_recall_paragraph_as_embeddings = test_recall_question_labels
#         y_test_recall_paragraph_as_label = test_recall_question_label_indx
#         x_test_recall_document_length = np.array([min(len(x), max_document_len) for x in x_test_recall_questions])
#     else:
#         x_test_recall_questions = test_recall_org_questions
#         y_test_recall_paragraph_as_embeddings = test_recall_question_labels
#         y_test_recall_paragraph_as_label = test_recall_question_label_indx
#         x_test_recall_document_length = tf.zeros(
#             [test_recall_org_questions.shape[0], test_recall_org_questions.shape[1]])
#
#     del test_recall_question_labels, test_recall_question_label_indx
#     print("x_test_recall_questions shape is {}".format(x_test_recall_questions.shape))
#     x_test_recall_baseline_question_embeddings = test_recall_org_questions
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (x_test_recall_questions,
#          x_test_recall_document_length,
#          x_test_recall_baseline_question_embeddings,
#          y_test_recall_paragraph_as_embeddings,
#          y_test_recall_paragraph_as_label))
#     dataset = dataset.batch(params.files["splitter"]["test_subset_size"])
#     dataset = dataset.map(parser_train)
#     # dataset = dataset.prefetch(1)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
# def train_recall_input_fn(base_data_path, params):
#     # TRAIN RECALL
#
#     train_recall_question_label_indx = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                          params.files['train_subset_recall'][
#                                                                              'question_labels']))
#     train_recall_question_label_indx = train_recall_question_label_indx.astype(int)
#
#     train_recall_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                    params.files['train_subset_recall'][
#                                                                        'question_embeddings']))
#
#     train_recall_question_labels = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                      params.files['train_subset_recall'][
#                                                                          'paragraph_embeddings']))
#
#     if params.model['model_type'].lower() == 'conv':
#         max_document_len = params.files['max_document_len']
#         x_train_recall_questions = UTIL.load_from_pickle(os.path.join(base_data_path,
#                                                                       params.files['train_subset_recall'][
#                                                                           'question_x_train_valid']))
#         y_train_recall_paragraph_as_embeddings = train_recall_question_labels
#         y_train_recall_paragraph_as_label = train_recall_question_label_indx
#         x_train_recall_document_length = np.array([min(len(x), max_document_len) for x in x_train_recall_questions])
#     else:
#         x_train_recall_questions = train_recall_org_questions
#         y_train_recall_paragraph_as_embeddings = train_recall_question_labels
#         y_train_recall_paragraph_as_label = train_recall_question_label_indx
#         x_train_recall_document_length = tf.zeros(
#             [train_recall_org_questions.shape[0], train_recall_org_questions.shape[1]])
#
#     del train_recall_question_labels, train_recall_question_label_indx
#     print("x_train_recall_questions shape is {}".format(x_train_recall_questions.shape))
#     x_train_recall_baseline_question_embeddings = train_recall_org_questions
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (x_train_recall_questions,
#          x_train_recall_document_length,
#          x_train_recall_baseline_question_embeddings,
#          y_train_recall_paragraph_as_embeddings,
#          y_train_recall_paragraph_as_label))
#     dataset = dataset.batch(params.files["splitter"]["train_subset_size"])
#     dataset = dataset.map(parser_train)
#     # dataset = dataset.prefetch(1)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
#
# def predict_input_fn(base_data_path, params):
#     prediction_org_questions = UTIL.load_embeddings(os.path.join(base_data_path,
#                                                                  params.files['prediction'][
#                                                                      'question_embeddings']))
#
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (prediction_org_questions,
#          ))
#     dataset = dataset.batch(params.model["batch_size"])
#     dataset = dataset.map(parser_estimate)
#     # dataset = dataset.prefetch(1)
#     iterator = dataset.make_one_shot_iterator()
#     return iterator.get_next()
#
#
#
# def train_input_fn(base_data_path, params):
#     """Train input function for the dataset.
#
#     Args:
#         base_data_path: (string) base path for all data
#         params: (Params) contains all the details of the execution including names of the data
#     """
#     print('train')
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
# def test_recall_input_fn(base_data_path, params):
#     """Test input function for the dataset.
#
#     Args:
#         base_data_path: (string) base path for all data
#         params: (Params) contains all the details of the execution including names of the data
#     """
#     print('test')
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
# def predict_input_fn(base_data_path, params):
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

