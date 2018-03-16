from tqdm import tqdm
import tensorflow as tf
from bilm.data import cached_path
import json
from bilm import Batcher, BidirectionalLanguageModel, TokenBatcher
from .data import UnicodeCharsVocabulary
from typing import List
import numpy as np
import h5py
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

DTYPE = 'float32'
DTYPE_INT = 'int64'
def weight_layers(name, bilm_ops, l2_coef=None,
                  use_top_only=False, do_layer_norm=False):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES 

    Input:
        name = a string prefix used for the trainable variable names
        bilm_ops = the tensorflow ops returned to compute internal
            representations from a biLM.  This is the return value
            from BidirectionalLanguageModel(...)(ids_placeholder)
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.
        use_top_only: if True, then only use the top layer.
        do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing

    Output:
        {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
        }
    '''
    def _l2_regularizer(weights):
        if l2_coef is not None:
            return l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    # Get ops for computing LM embeddings and mask
    lm_embeddings = bilm_ops['lm_embeddings']
    mask = bilm_ops['mask']

    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2
                                    ) / N
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )

        if use_top_only:
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
            # just the top layer
            sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
            # no regularization
            reg = 0.0
        else:
            W = tf.get_variable(
                '{}_ELMo_W'.format(name),
                shape=(n_lm_layers, ),
                initializer=tf.zeros_initializer,
                regularizer=_l2_regularizer,
                trainable=True,
            )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
            )
            # split LM layers
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
    
            # compute the weighted, normalized LM activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if do_layer_norm:
                    pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                else:
                    pieces.append(w * tf.squeeze(t, squeeze_dims=1))
            sum_pieces = tf.add_n(pieces)
    
            # get the regularizer 
            reg = [
                r for r in tf.get_collection(
                                tf.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_ELMo_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by gamma
        gamma = tf.get_variable(
            '{}_ELMo_gamma'.format(name),
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
        weighted_lm_layers = sum_pieces * gamma

        ret = {'weighted_op': weighted_lm_layers, 'regularization_op': reg}

    return ret

def empty_embedding(dims, reshaped=False) -> np.ndarray:
    if reshaped:
        return np.zeros((3, 0, dims))
    else:
        return np.zeros((0, dims))

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64

class ElmoEmbedder():
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 dims:int = 1024,
                 embedding_file = None) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        """
        self.options_file_path = cached_path(options_file)
        self.weight_file_path = cached_path(weight_file)
        with open(self.options_file_path, 'r') as fin:
            options = json.load(fin)
        self.max_word_length = options['char_cnn']['max_characters_per_token']
        self.dims = dims
        self.word_embedding_file = embedding_file
        # char file begin
        if self.word_embedding_file is None:
            self.ids_placeholder = tf.placeholder('int32', shape=(None, None, self.max_word_length))
            self.model = BidirectionalLanguageModel(self.options_file_path, self.weight_file_path)
        # char file end
        else:
            self.ids_placeholder = tf.placeholder('int32', shape=(None, None))
            self.model = BidirectionalLanguageModel(self.options_file_path, self.weight_file_path,
                                                    False, self.word_embedding_file)

        self.ops = self.model(self.ids_placeholder)
    def batch_to_vocs(self, batch: List[List[str]]):
        """
        Converts a batch of tokenized sentences to a voc as cached so that it can be used for Batchers

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            cached voc file path
        """
        self.file_path = cached_path('voc.txt')
        self.all_tokens = set(['<S>', '</S>', '<UNK>'])
        for _content in batch:
            for token in _content:
                if token.strip():
                    self.all_tokens.add(token)
        with open(self.file_path, 'w') as fout:
            fout.write('\n'.join(self.all_tokens))

        return self.file_path

    def list_to_token_embeddings(self,  batch: List[List[str]],
                                 inject_tfidf=False,
                                 tfidf_smooth_idf=False,
                                 tfidf_sublinear_tf=False,
                                 inject_mean_glove=False,
                                 required_tfidf_mean=False,
                                 glove_path=None,
                                 outfile_to_dump=None):
        '''
        Given an input vocabulary file, dump all the token embeddings to the
        outfile.  The result can be used as the embedding_weight_file when
        constructing a BidirectionalLanguageModel.
        '''

        #batcher = TokenBatcher(vocab_file)
        vocab = UnicodeCharsVocabulary(self.file_path, self.max_word_length)
        batcher = Batcher(self.file_path, self.max_word_length)
        embedding_op = self.ops['token_embeddings']
        n_tokens = vocab.size
        embed_dim = int(embedding_op.shape[2])
        embeddings = np.zeros((n_tokens, embed_dim), dtype=DTYPE)

        token_tfidf_weights = None
        if inject_tfidf:
            tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=tfidf_smooth_idf, sublinear_tf=tfidf_sublinear_tf)
            tfidf.fit(batch)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            token_tfidf_weights = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        if inject_mean_glove:
            glove_big = {}
            with open(glove_path, "rb") as infile:
                for line in infile:
                    parts = line.split()
                    word = parts[0].decode('utf-8')
                    if word in self.all_tokens:
                        nums = np.array(parts[1:], dtype=np.float32)
                        glove_big[word] = nums
            # dim = len(glove_big[next(iter(glove_big))])
            # np.array([
            #     np.mean([glove_big[w] for w in words if w in glove_big]
            #             or [np.zeros(dim)], axis=0)
            #     for words in batch
            # ])
        # token_glove_and_tfidf_weights = None
        # if inject_glove and inject_tfidf:
        #     return np.array([
        #         np.mean([glove_big[w] * token_tfidf_weights[w]
        #                  for w in words if w in glove_big] or
        #                 [np.zeros(dim)], axis=0)
        #         for words in batch
        #     ])


        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for k in tqdm(range(n_tokens)):
                token = vocab.id_to_word(k)
                char_ids = batcher.batch_sentences([[token]])[0, 1, :].reshape(
                    1, 1, -1)
                embeddings[k, :] = sess.run(
                    embedding_op, feed_dict={self.ids_placeholder: char_ids}
                )
                # if inject_tfidf and inject_mean_glove:
                #     if required_tfidf_mean:
                #         embeddings[k, :] = embeddings[k, :] * np.mean(token_tfidf_weights[token] * glove_big[token], axis=0)
                #     else:
                #         embeddings[k, :] = embeddings[k, :] * token_tfidf_weights[token] * np.mean(glove_big[token], axis=0)
                # elif inject_tfidf:
                #     embeddings[k, :] = embeddings[k, :] * token_tfidf_weights[token]
                # elif inject_mean_glove:
                #     embeddings[k, :] = embeddings[k, :] * np.mean(glove_big[token], axis=0)
                # else:
                #     pass


        with h5py.File(outfile_to_dump, 'w') as fout:
            ds = fout.create_dataset(
                'embedding', embeddings.shape, dtype='float32', data=embeddings
            )


        return embeddings, vocab._word_to_id

    def list_to_embeddings(self, batch: List[List[str]], slice=None, file_to_dump=None):
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        """
        elmo_embeddings = []

        if batch == [[]]:
            if slice is None:
                elmo_embeddings.append(empty_embedding(self.dims))
            else:
                if slice > 2:
                    raise ValueError('Slice can not be larger than 3')
                elmo_embeddings.append(empty_embedding(self.dims, True))
        else:
            batcher = Batcher(self.file_path, self.max_word_length)
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                for i, _contents in enumerate(tqdm(batch, total= len(batch))) :
                    #for _content in _contents:
                    #content_tokens_ = _contents.strip().split()
                    char_ids = batcher.batch_sentences([_contents])
                    _ops = sess.run(
                        self.ops, feed_dict={self.ids_placeholder: char_ids}
                    )
                    mask = _ops['mask']
                    lm_embeddings = _ops['lm_embeddings']
                    token_embeddings = _ops['token_embeddings']
                    lengths = _ops['lengths']
                    length = int(mask.sum())
                    if slice is None:
                        lm_embeddings_mean = np.apply_over_axes(np.mean, lm_embeddings[0], (0,1))
                    else:
                        lm_embeddings_mean = np.apply_over_axes(np.mean, lm_embeddings[0][slice], (0))
                    # if lm_embeddings.shape != (1,3,1,1024):
                    #     print('Index of batch:', i)
                    #     print('Contents:', _contents)
                    #     print('Content Tokens:', content_tokens_)
                    #     print('Shape:', lm_embeddings.shape)
                    #     print(10*'-')
                    elmo_embeddings.append(lm_embeddings_mean)

        return elmo_embeddings

    def list_to_embeddings_with_dump(self, batch: List[List[str]],outfile_to_dump=None,tfidfs=None):
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        """
        document_embeddings = []

        if batch == [[]]:
           raise ValueError('Batch should not be emppty')
        else:

            if self.word_embedding_file is None:
                batcher = Batcher(self.file_path, self.max_word_length)
            else:
                batcher = TokenBatcher(self.file_path)
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                ids_list = batcher.batch_sentences(batch)
                with h5py.File(outfile_to_dump, 'w') as fout:
                    for i, ids in enumerate(tqdm(ids_list, total=len(ids_list))):
                        _ops = sess.run(
                            self.ops, feed_dict={self.ids_placeholder: [ids]}
                        )
                        mask = _ops['mask']
                        lm_embeddings = _ops['lm_embeddings'][0,:]
                        token_embeddings = _ops['token_embeddings']
                        lengths = _ops['lengths']
                        length = int(mask.sum())
                        document_embeddings.append(lm_embeddings)
                        ds = fout.create_dataset(
                            '{}'.format(i),
                            lm_embeddings.shape, dtype='float32',
                            data=lm_embeddings
                        )
                document_embeddings = np.asarray(document_embeddings)
        return document_embeddings

    def list_to_embeddings_with_dump_(self, batch: List[List[str]], slice=None, outfile_to_dump=None,tfidfs=None):
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        """
        elmo_embeddings = []

        if batch == [[]]:
            if slice is None:
                elmo_embeddings.append(empty_embedding(self.dims))
            else:
                if slice > 2:
                    raise ValueError('Slice can not be larger than 3')
                elmo_embeddings.append(empty_embedding(self.dims, True))
        else:
            batcher = Batcher(self.file_path, self.max_word_length)
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                with h5py.File(outfile_to_dump, 'w') as fout:
                    for i, _contents in enumerate(tqdm(batch, total=len(batch))):
                        # for _content in _contents:
                        # content_tokens_ = _contents.strip().split()
                        char_ids = batcher.batch_sentences([_contents])
                        _ops = sess.run(
                            self.ops, feed_dict={self.ids_placeholder: char_ids}
                        )
                        mask = _ops['mask']
                        lm_embeddings = _ops['lm_embeddings']
                        token_embeddings = _ops['token_embeddings']
                        lengths = _ops['lengths']
                        length = int(mask.sum())
                        if slice is None:
                            lm_embeddings_mean = np.apply_over_axes(np.mean, lm_embeddings[0], (0, 1))
                        else:
                            lm_embeddings_mean = np.apply_over_axes(np.mean, lm_embeddings[0][slice], (0))
                        if tfidfs is not None:
                            lm_embeddings_mean = np.matmul(lm_embeddings_mean, np.reshape(tfidfs[i], (1, tfidfs[i].shape[0] )))
                        # if lm_embeddings.shape != (1,3,1,1024):
                        #     print('Index of batch:', i)
                        #     print('Contents:', _contents)
                        #     print('Content Tokens:', content_tokens_)
                        #     print('Shape:', lm_embeddings.shape)
                        #     print(10*'-')
                        elmo_embeddings.append(lm_embeddings_mean)
                        ds = fout.create_dataset(
                            '{}'.format(i),
                            lm_embeddings_mean[0].shape[1:], dtype='float32',
                            data=lm_embeddings_mean[0]
                        )

        return elmo_embeddings

    # def batch_to_ids(self, batch: List[List[str]]):
    #     """
    #     Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    #     (len(batch), max sentence length, max word length).
    #
    #     Parameters
    #     ----------
    #     batch : ``List[List[str]]``, required
    #         A list of tokenized sentences.
    #
    #     Returns
    #     -------
    #         A tensor of padded character ids.
    #     """
    #     instances = []
    #     for sentence in batch:
    #         tokens = [Token(token) for token in sentence]
    #         field = TextField(tokens,
    #                           {'character_ids': self.indexer})
    #         instance = Instance({"elmo": field})
    #         instances.append(instance)
    #
    #     dataset = Batch(instances)
    #     vocab = Vocabulary()
    #     dataset.index_instances(vocab)
    #     return dataset.as_tensor_dict()['elmo']['character_ids']


    # def embed_sentence(self, sentence: List[str]) -> numpy.ndarray:
    #     """
    #     Computes the ELMo embeddings for a single tokenized sentence.
    #
    #     Parameters
    #     ----------
    #     sentence : ``List[str]``, required
    #         A tokenized sentence.
    #
    #     Returns
    #     -------
    #     A tensor containing the ELMo vectors.
    #     """
    #
    #     return self.embed_batch([sentence])[0]
    #
    # def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
    #     """
    #     Computes the ELMo embeddings for a batch of tokenized sentences.
    #
    #     Parameters
    #     ----------
    #     batch : ``List[List[str]]``, required
    #         A list of tokenized sentences.
    #
    #     Returns
    #     -------
    #         A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
    #     """
    #     elmo_embeddings = []
    #
    #     # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
    #     # and return an empty embedding instead.
    #     if batch == [[]]:
    #         elmo_embeddings.append(empty_embedding())
    #     else:
    #         embeddings, mask = self.batch_to_embeddings(batch)
    #         for i in range(len(batch)):
    #             length = int(mask[i, :].sum())
    #             # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
    #             if length == 0:
    #                 elmo_embeddings.append(empty_embedding())
    #             else:
    #                 elmo_embeddings.append(embeddings[i, :, :length, :].data.cpu().numpy())
    #
    #     return elmo_embeddings
    #
    # def embed_sentences(self,
    #                     sentences: Iterable[List[str]],
    #                     batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[numpy.ndarray]:
    #     """
    #     Computes the ELMo embeddings for a iterable of sentences.
    #
    #     Parameters
    #     ----------
    #     sentences : ``Iterable[List[str]]``, required
    #         An iterable of tokenized sentences.
    #     batch_size : ``int``, required
    #         The number of sentences ELMo should process at once.
    #
    #     Returns
    #     -------
    #         A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
    #     """
    #     for batch in lazy_groups_of(iter(sentences), batch_size):
    #         yield from self.embed_batch(batch)
    #
    # def embed_file(self,
    #                input_file: IO,
    #                output_file_path: str,
    #                output_format: str = "all",
    #                batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    #     """
    #     Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
    #     The ELMo embeddings are written out in HDF5 format, where each sentences is saved in a dataset.
    #
    #     Parameters
    #     ----------
    #     input_file : ``IO``, required
    #         A file with one tokenized sentence per line.
    #     output_file_path : ``str``, required
    #         A path to the output hdf5 file.
    #     output_format : ``str``, optional, (default = "all")
    #         The embeddings to output.  Must be one of "all", "top", or "average".
    #     batch_size : ``int``, optional, (default = 64)
    #         The number of sentences to process in ELMo at one time.
    #     """
    #
    #     assert output_format in ["all", "top", "average"]
    #
    #     # Tokenizes the sentences.
    #     sentences = [line.strip() for line in input_file if line.strip()]
    #     split_sentences = [sentence.split() for sentence in sentences]
    #     # Uses the sentence as the key.
    #     embedded_sentences = zip(sentences, self.embed_sentences(split_sentences, batch_size))
    #
    #     logger.info("Processing sentences.")
    #     with h5py.File(output_file_path, 'w') as fout:
    #         for key, embeddings in Tqdm.tqdm(embedded_sentences):
    #             if key in fout.keys():
    #                 #logger.warning(f"Key already exists in {output_file_path}, skipping: {key}")
    #                 pass
    #             else:
    #                 if output_format == "all":
    #                     output = embeddings
    #                 elif output_format == "top":
    #                     output = embeddings[2]
    #                 elif output_format == "average":
    #                     output = numpy.average(embeddings, axis=0)
    #
    #                 fout.create_dataset(
    #                         str(key),
    #                         output.shape, dtype='float32',
    #                         data=output
    #                 )
    #     input_file.close()