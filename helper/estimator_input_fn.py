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
import h5py


class generator:
    def __init__(self, file, table_name='embeddings'):
        self.file = file
        self.table_name = table_name
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf[self.table_name]:
                yield im

class DataBuilder():
    """ DataBuilder class for all type of estimator input_fn
    """

    def __init__(self, base_path, params, verbose, load_with_file_path):
        self.params = params
        self.base_path = base_path
        self.verbose = verbose
        self.load_with_file_path = load_with_file_path
        self._init_data()

    def _read_dataset(self, file_path, embedding_dim, table_name='embeddings', data_type=tf.float32):
        if embedding_dim is not None:
            ds = tf.data.Dataset.from_generator(
                generator(file_path, table_name),
                data_type,
                tf.TensorShape([embedding_dim, ]))
        else:
            ds = tf.data.Dataset.from_generator(
                generator(file_path, table_name),
                data_type,
                tf.TensorShape([]))
        return ds

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

    def _load_data_path(self, data_type):
        source_labels_path = os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_LABELS']])
        source_indx_path = os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_IDX']])
        source_embeddings_path = os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_EMBEDDINGS']])
        target_embeddings_path = os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']])
        source_padded_path = os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_PADDED']])
        source_length_path = os.path.join(self.data_dir,
                                          self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_LENGTH']])
        print('Data Type: {}'.format(data_type))
        print('source_labels_path Path:{}'.format(source_labels_path))
        print('source_indx_path Path:{}'.format(source_indx_path))
        print('source_embeddings_path Path:{}'.format(source_embeddings_path))
        print('target_embeddings_path Path:{}'.format(target_embeddings_path))
        print('source_padded_path Path:{}'.format(source_padded_path))
        print('source_length_path Path:{}'.format(source_length_path))
        try:
            all_target_embeddings_path = os.path.join(self.data_dir,
                             self.params.files[data_type]["all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']])
        except:
            print('{} is not found for {} type'.format("all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS'], data_type))
            all_target_embeddings_path = None
        return source_labels_path, source_indx_path, source_embeddings_path, target_embeddings_path, all_target_embeddings_path, source_padded_path, source_length_path

    def _load_data(self, data_type):
        source_labels = UTIL.load_embeddings(os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_LABELS']])).astype(int)
        source_indx = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_IDX']])).astype(int)
        source_embeddings = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_SOURCE_EMBEDDINGS']]))
        target_embeddings = UTIL.load_embeddings(
            os.path.join(self.data_dir, self.params.files[data_type][self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']]))
        source_padded=None
        source_length=None
        try:
            all_target_embeddings = UTIL.load_embeddings(
                os.path.join(self.data_dir, self.params.files[data_type]["all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS']]))
        except:
            print('{} is not found for {} type'.format("all_" + self.KN_FILE_NAMES['KN_TARGET_EMBEDDINGS'], data_type))
            all_target_embeddings = None
        return source_labels, source_indx, source_embeddings, target_embeddings,all_target_embeddings, source_padded, source_length

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
            self._load_train_data(self.load_with_file_path)
            self._load_train_recall_data(self.load_with_file_path)
            self._load_test_recall_data(self.load_with_file_path)
            self._load_predict_data(self.load_with_file_path)
        else:
            if 'train' in verbose:
                self._load_train_data(self.load_with_file_path)
            if 'train_recall' in verbose:
                self._load_train_recall_data(self.load_with_file_path)
            if 'test_recall' in verbose:
                self._load_test_recall_data(self.load_with_file_path)
            if 'predict' in verbose:
                self._load_predict_data(self.load_with_file_path)

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
        if self.KN_FILE_NAMES['DIR'].lower().startswith('qu'):
            tokenized_sources = self._tokenized_questions
        else:
            tokenized_sources = self._tokenized_paragraphs
        indx_to_voc, self.voc_to_indx = UTIL.vocabulary_processor(tokenized_sources)
        self.vocab_size = len(self.voc_to_indx)
        self.params.files['vocab_size'] = self.vocab_size
        print('Vocab Size: %d' % self.vocab_size)
        # params.files['questions_vocab_size'] = vocab_size
        print('vocabulary is build on {}'.format(self.KN_FILE_NAMES['DIR']))

    def _obtain_tokenized_documents(self, source_indx):
        documents = []
        if self.load_with_file_path:
            source_indx = UTIL.load_embeddings(source_indx).astype(int)
        for indx in tqdm(source_indx):
            if self.KN_FILE_NAMES['DIR'].lower().startswith('qu'):
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

    """
    **********************************************
    TRAIN: START
    **********************************************
    """

    def train_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            if self.load_with_file_path:
                # to get the size of the data
                _train_source_embeddings = self._read_dataset(self._train_source_embeddings,self.params.files['max_document_len'], data_type=tf.int32)
                _train_source_embeddings_lengths = self._read_dataset(self._train_source_embeddings_lengths,
                                                       None, data_type=tf.int32)
                _train_baseline_source_embeddings = self._read_dataset(self._train_baseline_source_embeddings,self.params.files['pre_trained_files']['embedding_dim'])
                _train_target_embeddings = self._read_dataset(self._train_target_embeddings,
                                                                       self.params.files['pre_trained_files'][
                                                                           'embedding_dim'])
                _train_source_labels = self._read_dataset(self._train_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_train_source_embeddings, _train_source_embeddings_lengths, _train_baseline_source_embeddings, _train_target_embeddings, _train_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((self._train_source_embeddings,
                                                             self._train_source_embeddings_lengths,
                                                             self._train_baseline_source_embeddings,
                                                             self._train_target_embeddings,
                                                             self._train_source_labels))
            dataset = dataset.map(self._parser_conv)
        else:
            if self.load_with_file_path:
                _train_source_embeddings = self._read_dataset(self._train_source_embeddings,self.params.files['pre_trained_files'][
                                                                           'embedding_dim'])
                _train_target_embeddings = self._read_dataset(self._train_target_embeddings,
                                                                       self.params.files['pre_trained_files'][
                                                                           'embedding_dim'])
                _train_source_labels = self._read_dataset(self._train_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_train_source_embeddings, _train_target_embeddings, _train_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices((self._train_source_embeddings,
                                                              self._train_target_embeddings,
                                                              self._train_source_labels))
            dataset = dataset.map(self._parser_non_conv)
        # print('_train_source_embeddings Path:{}'.format(self._train_source_embeddings))
        # print('_train_target_embeddings Path:{}'.format(self._train_target_embeddings))
        # print('_train_source_labels Path:{}'.format(self._train_source_labels))
        if self.params.model["shuffle"]:
            dataset = dataset.shuffle(buffer_size=self._temp_train_source_labels.shape[0])
        if self.params.loss['version'] in [4,5]:
            unique, counts = np.unique(self._temp_train_source_labels, return_counts=True)
            print('class size is {}'.format(unique.shape[0]))
            target_list_np = np.random.choice(self._temp_train_source_labels.shape[0], self.params.model["batch_size"], replace=False)
            for i, t in enumerate(target_list_np):
                if i == 0:
                    dataset_ = dataset.filter(lambda features, labels: tf.equal(labels['target_labels'], t)).take(1)
                else:
                    dataset_ = dataset_.concatenate(
                        dataset.filter(lambda features, labels: tf.equal(labels['target_labels'], t)).take(1))
            dataset = dataset_
        dataset = dataset.batch(self.params.model["batch_size"])
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        #def helper.globalizer.train_input_fn():return dataset
        return iterator.get_next()

    def _load_train_data(self, load_with_file_path=False):
        if load_with_file_path:
            ## LOAD WITH FILE PATHS
            self._train_source_labels, self._train_source_indx, self._train_source_embeddings, self._train_target_embeddings, self._train_all_target_embeddings, self._train_source_padded, self._train_source_length = self._load_data_path('train_loss')
            self._temp_train_source_labels = UTIL.load_embeddings(self._train_source_labels)
        else:
            ## LOAD WITH ACTUAL DATA
            self._train_source_labels, self._train_source_indx, self._train_source_embeddings, self._train_target_embeddings, self._train_all_target_embeddings, self._train_source_padded, self._train_source_length = self._load_data(
                'train_loss')
            self._temp_train_source_labels = self._train_source_labels
        self._train_baseline_source_embeddings = self._train_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._train_source_indx)
            self._train_source_embeddings, self._train_source_embeddings_lengths = self._pad_documents(tokenized_documents)
            if load_with_file_path:
                ## SAVE self._train_source_embeddings, self._train_source_embeddings_lengths SO THAT IT CAN BE RELOADED FROM FILE
                UTIL.dump_embeddings(self._train_source_embeddings, self._train_source_padded)
                UTIL.dump_embeddings(self._train_source_embeddings_lengths, self._train_source_length, dtype="int32")
                self._train_source_embeddings, self._train_source_embeddings_lengths = self._train_source_padded, self._train_source_length

        else:
            self._train_source_embeddings_lengths = np.zeros([self._temp_train_source_labels.shape[0], 1])
            if load_with_file_path:
                ## SAVE self._train_source_embeddings, self._train_source_embeddings_lengths SO THAT IT CAN BE RELOADED FROM FILE
                UTIL.dump_embeddings(self._train_source_embeddings_lengths, self._train_source_length, dtype="int32")
                self._train_source_embeddings_lengths = self._train_source_length
        # print('_train_source_labels Path:{}'.format(self._train_source_labels))
        # print('_train_source_indx Path:{}'.format(self._train_source_indx))
        # print('_train_source_embeddings Path:{}'.format(self._train_source_embeddings))
        # print('_train_target_embeddings Path:{}'.format(self._train_target_embeddings))
        # print('_train_all_target_embeddings Path:{}'.format(self._train_all_target_embeddings))
        # print('_train_source_embeddings Path:{}'.format(self._train_source_embeddings))
        # print('_train_source_embeddings_lengths Path:{}'.format(self._train_source_embeddings_lengths))
    """
    **********************************************
    TRAIN: END
    **********************************************
    """
    # ----------------------------------------------------------------------------------
    """
    **********************************************
    TRAIN RECALL: START
    **********************************************
    """
    def _load_train_recall_data(self, load_with_file_path=False):
        if load_with_file_path:
            self._train_recall_source_labels, self._train_recall_source_indx, self._train_recall_source_embeddings, self._train_recall_target_embeddings, self._train_recall_all_target_embeddings, self._train_recall_source_padded, self._train_recall_source_length = self._load_data_path(
                'train_subset_recall')
            self._temp_train_recall_source_labels = UTIL.load_embeddings(self._train_recall_source_labels)
        else:
            self._train_recall_source_labels, self._train_recall_source_indx, self._train_recall_source_embeddings, self._train_recall_target_embeddings, self._train_recall_all_target_embeddings, self._train_recall_source_padded, self._train_recall_source_length  = self._load_data('train_subset_recall')
            self._temp_train_recall_source_labels = self._train_recall_source_labels
        self._train_recall_baseline_source_embeddings = self._train_recall_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._train_recall_source_indx)
            self._train_recall_source_embeddings, self._train_recall_source_embeddings_lengths = self._pad_documents(
                tokenized_documents)
            if load_with_file_path:
                UTIL.dump_embeddings(self._train_recall_source_embeddings, self._train_recall_source_padded)
                UTIL.dump_embeddings(self._train_recall_source_embeddings_lengths,  self._train_recall_source_length, dtype="int32")
                self._train_recall_source_embeddings, self._train_recall_source_embeddings_lengths = self._train_recall_source_padded, self._train_recall_source_length
        else:
            self._train_recall_source_embeddings_lengths = np.zeros(
                [self._temp_train_recall_source_labels.shape[0], 1])
            if load_with_file_path:
                UTIL.dump_embeddings(self._train_recall_source_embeddings_lengths, self._train_recall_source_length,
                                     dtype="int32")
                self._train_recall_source_embeddings_lengths = self._train_recall_source_length
    def train_recall_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            if self.load_with_file_path:
                _train_recall_source_embeddings = self._read_dataset(self._train_recall_source_embeddings,self.params.files['max_document_len'], data_type=tf.int32)
                _train_recall_source_embeddings_lengths = self._read_dataset(self._train_recall_source_embeddings_lengths,
                                                       None, data_type=tf.int32)
                _train_recall_baseline_source_embeddings = self._read_dataset(self._train_recall_baseline_source_embeddings,self.params.files['pre_trained_files']['embedding_dim'])
                _train_recall_target_embeddings = self._read_dataset(self._train_recall_target_embeddings,
                                                                       self.params.files['pre_trained_files'][
                                                                           'embedding_dim'])
                _train_recall_source_labels = self._read_dataset(self._train_recall_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_train_recall_source_embeddings, _train_recall_source_embeddings_lengths, _train_recall_baseline_source_embeddings, _train_recall_target_embeddings, _train_recall_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                        (self._train_recall_source_embeddings,
                        self._train_recall_source_embeddings_lengths,
                        self._train_recall_baseline_source_embeddings,
                        self._train_recall_target_embeddings,
                        self._train_recall_source_labels))

            dataset = dataset.map(self._parser_conv)
        else:
            if self.load_with_file_path:
                _train_recall_source_embeddings = self._read_dataset(self._train_recall_source_embeddings,
                                                                     self.params.files['pre_trained_files'][
                                                                         'embedding_dim']
                                                                     )

                _train_recall_target_embeddings = self._read_dataset(self._train_recall_target_embeddings,
                                                                     self.params.files['pre_trained_files'][
                                                                         'embedding_dim'])
                _train_recall_source_labels = self._read_dataset(self._train_recall_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_train_recall_source_embeddings, _train_recall_target_embeddings, _train_recall_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self._train_recall_source_embeddings,
                     self._train_recall_target_embeddings,
                     self._train_recall_source_labels))
            dataset = dataset.map(self._parser_non_conv)

        dataset = dataset.batch(self.params.files["splitter"]["train_subset_size"])
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        # return dataset #iterator.get_next()
        # global train_recall_input_fn
        #helper.globalizer.train_recall_input_fn  = dataset
        return iterator.get_next()
    """
    **********************************************
    TRAIN RECALL: END
    **********************************************
    """
    #----------------------------------------------------------------------------------
    """
    **********************************************
    TEST RECALL: START
    **********************************************
    """
    def test_recall_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            if self.load_with_file_path:
                _test_recall_source_embeddings = self._read_dataset(self._test_recall_source_embeddings,self.params.files['max_document_len'], data_type=tf.int32)
                _test_recall_source_embeddings_lengths = self._read_dataset(self._test_recall_source_embeddings_lengths,
                                                       None, data_type=tf.int32)
                _test_recall_baseline_source_embeddings = self._read_dataset(self._test_recall_baseline_source_embeddings,self.params.files['pre_trained_files']['embedding_dim'])
                _test_recall_target_embeddings = self._read_dataset(self._test_recall_target_embeddings,
                                                                       self.params.files['pre_trained_files'][
                                                                           'embedding_dim'])
                _test_recall_source_labels = self._read_dataset(self._test_recall_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_test_recall_source_embeddings, _test_recall_source_embeddings_lengths, _test_recall_baseline_source_embeddings, _test_recall_target_embeddings, _test_recall_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self._test_recall_source_embeddings,
                     self._test_recall_source_embeddings_lengths,
                     self._test_recall_baseline_source_embeddings,
                     self._test_recall_target_embeddings,
                     self._test_recall_source_labels))
            dataset = dataset.map(self._parser_conv)
        else:
            if self.load_with_file_path:
                _test_recall_source_embeddings = self._read_dataset(self._test_recall_source_embeddings,
                                                                    self.params.files['pre_trained_files'][
                                                                        'embedding_dim'])
                _test_recall_target_embeddings = self._read_dataset(self._test_recall_target_embeddings,
                                                                    self.params.files['pre_trained_files'][
                                                                        'embedding_dim'])
                _test_recall_source_labels = self._read_dataset(self._test_recall_source_labels,
                                                          None,data_type=tf.int64)
                dataset = tf.data.Dataset.zip((_test_recall_source_embeddings, _test_recall_target_embeddings,
                                               _test_recall_source_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self._test_recall_source_embeddings,
                     self._test_recall_target_embeddings,
                     self._test_recall_source_labels))
            dataset = dataset.map(self._parser_non_conv)

        # print('_train_source_embeddings Path:{}'.format(self._test_recall_source_embeddings))
        # print('_train_target_embeddings Path:{}'.format(self._test_recall_target_embeddings))
        # print('_train_source_labels Path:{}'.format(self._test_recall_source_labels))
        dataset = dataset.batch(self.params.files["splitter"]["test_subset_size"])

        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        #return dataset #iterator.get_next()
        # global test_recall_input_fn
        return iterator.get_next()
    def _load_test_recall_data(self, load_with_file_path=False):
        if load_with_file_path:
            self._test_recall_source_labels, self._test_recall_source_indx, self._test_recall_source_embeddings, self._test_recall_target_embeddings, self._test_recall_all_target_embeddings, self._test_recall_source_padded, self._test_recall_source_length = self._load_data_path(
                'test_subset_recall')
            self._temp_test_recall_source_labels = UTIL.load_embeddings(self._test_recall_source_labels)
        else:
            self._test_recall_source_labels, self._test_recall_source_indx, self._test_recall_source_embeddings, self._test_recall_target_embeddings, self._test_recall_all_target_embeddings, self._test_recall_source_padded, self._test_recall_source_length  = self._load_data('test_subset_recall')
            self._temp_test_recall_source_labels = self._test_recall_source_labels
        self._test_recall_baseline_source_embeddings = self._test_recall_source_embeddings
        if self.params.model['model_type'].lower() == 'conv':
            tokenized_documents = self._obtain_tokenized_documents(self._test_recall_source_indx)
            self._test_recall_source_embeddings, self._test_recall_source_embeddings_lengths = self._pad_documents(tokenized_documents)
            if load_with_file_path:
                UTIL.dump_embeddings(self._test_recall_source_embeddings, self._test_recall_source_padded)
                UTIL.dump_embeddings(self._test_recall_source_embeddings_lengths,  self._test_recall_source_length, dtype="int32")
                self._test_recall_source_embeddings, self._test_recall_source_embeddings_lengths = self._test_recall_source_padded, self._test_recall_source_length
        else:
            self._test_recall_source_embeddings_lengths = np.zeros(
                [self._temp_test_recall_source_labels.shape[0], 1])
            if load_with_file_path:
                UTIL.dump_embeddings(self._test_recall_source_embeddings_lengths, self._test_recall_source_length,
                                     dtype="int32")
                self._test_recall_source_embeddings_lengths = self._test_recall_source_length
        # print('_test_recall_source_labels Path:{}'.format(self._test_recall_source_labels))
        # print('_test_recall_source_indx Path:{}'.format(self._test_recall_source_indx))
        # print('_test_recall_source_embeddings Path:{}'.format(self._test_recall_source_embeddings))
        # print('_test_recall_target_embeddings Path:{}'.format(self._test_recall_target_embeddings))
        # print('_test_recall_all_target_embeddings Path:{}'.format(self._test_recall_all_target_embeddings))
        # print('_test_recall_source_embeddings Path:{}'.format(self._test_recall_source_embeddings))
        # print('_test_recall_source_embeddings_lengths Path:{}'.format(self._test_recall_source_embeddings_lengths))
    """
    **********************************************
    TEST RECALL: END
    **********************************************
    """
    # ----------------------------------------------------------------------------------
    """
    **********************************************
    PREDICT: START
    **********************************************
    """
    def predict_input_fn(self):
        if self.params.model['model_type'].lower() == 'conv':
            if self.load_with_file_path:
                _source_embeddings = self._read_dataset(self._source_embeddings,
                                                                    self.params.files['max_document_len'],
                                                                    data_type=tf.int32)
                _source_embeddings_lengths = self._read_dataset(self._source_embeddings_lengths,
                                                                    None,tf.int32)
                dataset = tf.data.Dataset.zip((_source_embeddings, _source_embeddings_lengths))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self._source_embeddings,
                     self._source_embeddings_lengths))
            dataset = dataset.map(self._parser_estimate_conv)
        else:
            if self.load_with_file_path:
                _source_embeddings = self._read_dataset(self._source_embeddings,
                                                        self.params.files['pre_trained_files'][
                                                            'embedding_dim']
                                                        )
                dataset = tf.data.Dataset.zip((_source_embeddings))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self._source_embeddings))
            dataset = dataset.map(self._parser_estimate_non_conv)

        dataset = dataset.batch(self.params.model["batch_size"])
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        # global predict_input_fn
        # predict_input_fn = dataset
        return dataset #iterator.get_next()
    def _load_predict_data(self, load_with_file_path=False):
        if load_with_file_path:
            self._source_embeddings = os.path.join(self.base_path, self.params.files['prediction']['source_embeddings'])
            self._source_padded = os.path.join(self.base_path, self.params.files['prediction']['source_padded'])
            self._source_length = os.path.join(self.base_path, self.params.files['prediction']['source_length'])
            self._temp_source_embeddings = UTIL.load_embeddings(self._source_embeddings)
        else:
            self._source_embeddings = UTIL.load_embeddings(
                os.path.join(self.base_path, self.params.files['prediction']['source_embeddings']))
            self._source_padded =None
            self._source_length = None
            self._temp_source_embeddings = self._source_embeddings
        if self.KN_FILE_NAMES['DIR'].lower().startswith('qu'):
            tokenized_documents = self._tokenized_questions
        else:
            tokenized_documents = self._tokenized_paragraphs
        if self.params.model['model_type'].lower() == 'conv':
            self._source_embeddings, self._source_embeddings_lengths = self._pad_documents(
                tokenized_documents)
            if load_with_file_path:
                UTIL.dump_embeddings(self._source_embeddings, self._source_padded)
                UTIL.dump_embeddings(self._source_embeddings_lengths, self._source_length,
                                     dtype="int32")
                self._source_embeddings,self._source_embeddings_lengths  = self._source_padded, self._source_length
        else:
            self._source_embeddings_lengths = np.zeros(
                [self._temp_source_embeddings.shape[0], 1])
            if load_with_file_path:
                UTIL.dump_embeddings(self._source_embeddings_lengths, self._source_length,
                                     dtype="int32")
                self._source_embeddings_lengths = self._source_length
    """
    **********************************************
    PREDICT: END
    **********************************************
    """





# -------------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


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