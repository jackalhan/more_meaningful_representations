import os
import sys
import numpy as np
import h5py
import spacy
import random
from tqdm import tqdm
import tensorflow_hub as hub
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
import argparse
#from allennlp.commands.elmo import ElmoEmbedder
from gensim.summarization.bm25 import get_bm25_weights
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math
import six
import array
import io
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', default=4, type=int, help="-1: no windows, else should be selected from the given range 1-512. it should be at least 1 lower than truncate_length")
    parser.add_argument('--data_path', help="as a master path to locate all files and older underneath")
    parser.add_argument('--is_read_contents_from_squad_format', type=UTIL.str2bool)
    parser.add_argument('--max_tokens', default=-1, type=int, help='Whether to limit the number of tokens per context')
    parser.add_argument('--dataset_path', type=str, help='whether squad formatted json file or a folder that has individual files')
    parser.add_argument('--pre_generated_embeddings_path', type=str,
                        help='for embeddings such as bert, pre embeddings needs to be located a folder and a folder needs to be specified in this arg. source, destination folders needs to be under that folder')
    parser.add_argument('--conc_layers', default=None, help='whether to concatenate all specified layers or None')
    parser.add_argument('--ind_layers', default=None,
                        help='whether to create individual representations for specified layer or None, 0 for token layer')
    parser.add_argument('--is_inject_idf', default=True, type=UTIL.str2bool, help="whether inject idf or not to the weights")
    parser.add_argument('--is_averaged_token', default=True, type=UTIL.str2bool,
                        help="For FCN models, it must be set to True, for convolution kernels, it must be set to False, but the generated files will be very large")
    parser.add_argument('--document_source_partition_size', default=10000, type=int,
                        help="size of partition to handle documents")
    parser.add_argument('--document_source_index', default=0, type=int,
                        help="last document index processed so that parititon would continue from that index on")
    parser.add_argument('--document_destination_partition_size', default=2000, type=int,
                        help="size of partition to handle documents")
    parser.add_argument('--document_destination_index', default=0, type=int,
                        help="last document index processed so that parititon would continue from that index on")
    parser.add_argument('--test_size', default=None,
                        help="question, paragraph sizes")
    parser.add_argument('--embedding_type', default='elmo', type=str,
                        help="elmo, bert, glove")
    parser.add_argument('--document_verbose', default=3, type=int,
                        help="1:question, 2:paragraph, 3:document_and_paragraph")
    parser.add_argument('--is_stack_all', default=True, type=UTIL.str2bool,
                        help="stack all data into one big file")
    parser.add_argument('--token_verbose', default=None, type=int,
                        help="verbose: None, just tokenize on sent, 1, "
                             "lowercase and tokenize on sent 2, "
                             "lemmatize(lowercased) and tokenize on sent, "
                             "3, tokenize and remove stopwords on sent, "
                             "4, lowercase, tokenize and remove stopwords on sent, "
                             "5, lemmatize(lowercased), tokenize and remove stopwords on sent")

    return parser


glove_embeddings = None
fasttext_embeddings = None
tfidf_transformer=None
bm25_embeddings = None
#thanks to gensim!!! since it does not provide fit/transform structure, I need to trace the index of the corpus in order to reach the specified document
bm25_index=None

def get_file_names(path, file_name_splitter, bert_extension):
    bert_embeddings_file_names = []
    for name in [name for name in os.listdir(path)
                  if name.endswith(bert_extension)]:
        _names = name.rpartition('.')[0].split(file_name_splitter)[3:6]
        _names.remove('to')
        item = [int(index) for index in _names] + [name]
        bert_embeddings_file_names.append((item, os.path.join(path,name)))
    bert_embeddings_file_names.sort()
    return bert_embeddings_file_names

def find_file_name(index, file_names):
    for file_name, file_path in file_names:
        if file_name[0] <= index <= file_name[1]:
            return file_name[2], file_path, file_name[0] - index


def finalize_embeddings(document_embeddings, file_path, last_index):
    if args.is_averaged_token:
        document_embeddings = np.asarray(document_embeddings)
    else:
        document_embeddings = pad(document_embeddings, args.max_tokens)

    print('Embeddings are completed with a shape of {}'.format(document_embeddings.shape))
    UTIL.dump_embeddings(document_embeddings, file_path)
    print('Embeddings are dumped with a name of {}'.format(file_path))
    print('Last_index is {}'.format(last_index))
    print('*' * 15)

def stack_partitioned_embeddings(path, file_name_extension, name_prefix, partition_size, is_averaged_token):
    names=[]
    for name in [name for name in os.listdir(path)
                 if name.startswith(name_prefix)]:
        _name = int(name.rpartition('_')[2].rpartition('.')[0])
        names.append(_name)
    names.sort()
    print('File names: {}'.format(names))
    partition_size = int(partition_size / 10)
    range_size = math.ceil(len(names) / (partition_size))
    for part in range(0, range_size):
        embeddings = None
        start = part * partition_size
        end = start + partition_size
        for name in tqdm(names[start:end]):
            name_path = os.path.join(path, name_prefix + str(name) + '.hdf5')
            embedding = UTIL.load_embeddings(name_path)
            if not is_averaged_token:
                embedding = np.expand_dims(embedding, axis=0)
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = np.vstack((embeddings, embedding))
        UTIL.dump_embeddings(embeddings, os.path.join(path, 'all_embeddings_'+file_name_extension + '_part__{}.hdf5'.format(part)))

    embeddings = None
    for part in range(0, range_size):
        embedding = UTIL.load_embeddings(os.path.join(path, 'all_embeddings_'+file_name_extension + '_part__{}.hdf5'.format(part)))
        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = np.vstack(
                (embeddings, embedding))

    UTIL.dump_embeddings(embeddings, os.path.join(path,
                                                      'all_embeddings_' + file_name_extension + '.hdf5'))

    for part in range(0, range_size):
        os.remove(os.path.join(path, 'all_embeddings_'+file_name_extension + '_part__{}.hdf5'.format(part)))

    print("Embeddings are getting dumped {}".format(embeddings.shape))



def partitionally_generate_generic_embeddings(documents_as_tokens, path, args, conc_layers, ind_layers, partition_size, last_index, file_name_extension, token2idfweight, embedding_function):
    document_embeddings = []

    # if args.ind_layer is not None:
    #     if ind_layer == 0:
    #         elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    #
    # else:
    name_prefix = 'partitioned_' +  file_name_extension + '_'
    for partition_counter in range(0, len(documents_as_tokens), partition_size):
        partition_number = partition_counter + partition_size
        file_path = os.path.join(path, name_prefix + str(partition_counter) + '.hdf5')
        if partition_number <= last_index: #10000 <= 5000 - False
            continue
        if os.path.exists(file_path):
            continue
            #raise Exception('Check the last index document since it is already processed in {}'.format(file_path))
        start_index = last_index # 5000
        document_embeddings = []
        print('Partition Counter:{}'.format(partition_counter))
        print('Partition Size:{}'.format(partition_size))
        print('Partition Number:{}'.format(partition_number))
        print('Start Index:{}'.format(start_index))
        try:
            for doc_indx, elmo_embedding in tqdm(enumerate(embedding_function(documents_as_tokens[start_index:]), start_index)):
                last_index = doc_indx # 5000
                global bm25_index
                bm25_index = doc_indx
                if last_index < partition_number: # 5000 < 5000 True
                    if  args.ind_layers is not None:
                        token_embeddings = np.array(
                            [l for l_indx, l in enumerate(reversed(elmo_embedding), 1) if  l_indx * -1 in ind_layers])
                        token_embeddings = np.average(token_embeddings, axis=0)
                    else:
                        token_embeddings = np.concatenate(
                            [l for l_indx, l in enumerate(reversed(elmo_embedding), 1) if l_indx * -1 in conc_layers], axis=1)

                    if args.is_inject_idf:
                        injected_idf_embeddings = []
                        for token in documents_as_tokens[doc_indx]:
                            injected_idf_embeddings.append(token2idfweight[token])
                        injected_idf_embeddings = np.asarray(injected_idf_embeddings).reshape(-1,1)
                        token_embeddings = np.multiply(token_embeddings, injected_idf_embeddings)
                    if args.is_averaged_token:
                        token_embeddings = np.mean(token_embeddings, axis=0)
                    document_embeddings.append(token_embeddings)
                else:
                    finalize_embeddings(document_embeddings, file_path, last_index)
                    document_embeddings = []
                    break
            if len(document_embeddings) != 0:
                finalize_embeddings(document_embeddings, file_path, last_index)
        except Exception as ex:
            print('last_index:{}'.format(last_index))
            raise Exception(ex)
    return name_prefix
def individually_generate_generic_embeddings(documents_as_tokens, path, args, conc_layers, ind_layers, partition_size, last_index, file_name_extension, token2idfweight, embedding_function):

    name_prefix = 'individually_' + file_name_extension + '_'
    try:
        start_index = last_index
        for doc_indx, elmo_embedding in tqdm(enumerate(embedding_function(documents_as_tokens[start_index:]), start_index)):
            last_index = doc_indx # 5000
            global bm25_index
            bm25_index = doc_indx
            file_path = os.path.join(path,
                                     name_prefix + str(doc_indx) + '.hdf5')
            if  args.ind_layers is not None:
                token_embeddings = np.array(
                    [l for l_indx, l in enumerate(reversed(elmo_embedding), 1) if  l_indx * -1 in ind_layers])
                token_embeddings = np.average(token_embeddings, axis=0)
            else:
                token_embeddings = np.concatenate(
                    [l for l_indx, l in enumerate(reversed(elmo_embedding), 1) if l_indx * -1 in conc_layers], axis=1)

            if args.is_inject_idf:
                injected_idf_embeddings = []
                for token in documents_as_tokens[doc_indx]:
                    injected_idf_embeddings.append(token2idfweight[token])
                injected_idf_embeddings = np.asarray(injected_idf_embeddings).reshape(-1,1)
                token_embeddings = np.multiply(token_embeddings, injected_idf_embeddings)
            token_embeddings = np.expand_dims(token_embeddings, axis=0)
            token_embeddings = pad(token_embeddings, args.max_tokens)
            UTIL.dump_embeddings(token_embeddings, file_path)
    except Exception as ex:
        print('last_index:{}'.format(last_index))
        raise Exception(ex)
    return name_prefix

def embed_with_glove(documents_as_tokens):
    for doc_indx, tokens in enumerate(documents_as_tokens):
        token_embeddings = None
        for token in tokens:
            token_embedding = glove_embeddings[token]
            token_embedding = np.expand_dims(token_embedding, axis=0)
            if token_embeddings is None:
                token_embeddings = token_embedding
            else:
                token_embeddings = np.append(token_embeddings, token_embedding, axis=0)
        print('*' * 100)
        print('*' * 100)
        print('embed_with_glove')
        print('*' * 20)
        print('doc_indx:{}'.format(doc_indx))
        print('len of tokens:{}'.format(len(tokens)))
        print('token embeddings:{}'.format(token_embeddings.shape))
        print('tokens: {}'.format(tokens))
        print('=' * 20)
        yield np.expand_dims(token_embeddings, axis=0)

def embed_with_fasttext(documents_as_tokens):
    for doc_indx, tokens in enumerate(documents_as_tokens):
        token_embeddings = None
        for token in tokens:
            token_embedding = fasttext_embeddings[token]
            token_embedding = np.expand_dims(token_embedding, axis=0)
            if token_embeddings is None:
                token_embeddings = token_embedding
            else:
                token_embeddings = np.append(token_embeddings, token_embedding, axis=0)
        print('*' * 100)
        print('*' * 100)
        print('embed_with_fasttext')
        print('*' * 20)
        print('doc_indx:{}'.format(doc_indx))
        print('len of tokens:{}'.format(len(tokens)))
        print('token embeddings:{}'.format(token_embeddings.shape))
        print('tokens: {}'.format(tokens))
        print('=' * 20)
        yield np.expand_dims(token_embeddings, axis=0)

def embed_with_bert(documents_as_tokens, start_indx, file_names, window_length):

    checkpoint=None
    for doc_indx, tokens in enumerate(documents_as_tokens, start_indx):
        bert_index = doc_indx + 1
        file_name, file_path, remaining_index_to_pass_this_file = find_file_name(bert_index, file_names)
        if remaining_index_to_pass_this_file >= 0:
            jsons = UTIL.load_bert_jsons_from_single_file(file_path)
            if doc_indx > 0:
                checkpoint = doc_indx
        if checkpoint is not None:
            doc_indx = doc_indx - checkpoint
        # checkpoint is required since actual document index is different than the same document index for partitioned pre generated embedding files.
        token_embeddings = None
        for line_index, json in UTIL.reversedEnumerate(jsons[doc_indx]):
            # 0 and -1 token indexes belong to [CLS, SEP] we are ignoring them.
            json['features'].pop(0)
            json['features'].pop(-1)

            # filter out the non-contributional tokens from the list.
            features = [x for x in json['features'] if not x['token'].startswith("##")]
            for feature_index, feature in UTIL.reversedEnumerate(features):
                if line_index > 0 and feature_index < window_length:
                    # print(feature['token'])
                    continue

                token_embedding = np.array(
                        [l['values'] for l in feature['layers']])
                token_embedding = np.expand_dims(token_embedding, axis=1)
                if token_embeddings is None:
                    token_embeddings = token_embedding
                else:
                    token_embeddings = np.append(token_embeddings, token_embedding, axis=1)
        print('*' * 100)
        print('*' * 100)
        print('embed_with_bert')
        print('*' * 20)
        print('Checkpoint:{}'.format(checkpoint))
        print('doc_indx:{}'.format(doc_indx))
        print('len of tokens:{}'.format(len(tokens)))
        print('token embeddings:{}'.format(token_embeddings.shape))
        print('tokens: {}'.format(tokens))
        print('=' * 20)
        yield token_embeddings
def convert_bert_tokens(documents_as_tokens, file_names, window_length):

    checkpoint=None
    all_new_tokens = []
    for doc_indx, tokens in tqdm(enumerate(documents_as_tokens)):
        bert_index = doc_indx + 1
        file_name, file_path, remaining_index_to_pass_this_file = find_file_name(bert_index, file_names)
        if remaining_index_to_pass_this_file >= 0:
            jsons = UTIL.load_bert_jsons_from_single_file(file_path)
            if doc_indx > 0:
                checkpoint = doc_indx
        if checkpoint is not None:
            doc_indx = doc_indx - checkpoint
        # checkpoint is required since actual document index is different than the same document index for partitioned pre generated embedding files.
        new_tokens = []
        for line_index, json in UTIL.reversedEnumerate(jsons[doc_indx]):
            # 0 and -1 token indexes belong to [CLS, SEP] we are ignoring them.
            json['features'].pop(0)
            json['features'].pop(-1)

            # filter out the non-contributional tokens from the list.
            features = [x for x in json['features'] if not x['token'].startswith("##")]
            for feature_index, feature in UTIL.reversedEnumerate(features):
                if line_index > 0 and feature_index < window_length:
                    # print(feature['token'])
                    continue
                new_tokens.append(feature['token'])
        all_new_tokens.append(new_tokens)

    return all_new_tokens
def partitionally_generate_bert_embeddings(documents_as_tokens, path, args, conc_layers, ind_layers, partition_size, last_index, file_name_extension, token2idfweight, file_names):

    # if args.ind_layer is not None:
    #     if ind_layer == 0:
    #         elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    #
    # else:
    name_prefix = 'partitioned_' +  file_name_extension + '_'
    for partition_counter in range(0, len(documents_as_tokens), partition_size):
        partition_number = partition_counter + partition_size
        file_path = os.path.join(path, name_prefix + str(partition_counter) + '.hdf5')
        if partition_number <= last_index: #10000 <= 5000 - False
            continue
        if os.path.exists(file_path):
            continue
            #raise Exception('Check the last index document since it is already processed in {}'.format(file_path))
        start_index = last_index # 5000
        document_embeddings = []
        print('Partition Counter:{}'.format(partition_counter))
        print('Partition Size:{}'.format(partition_size))
        print('Partition Number:{}'.format(partition_number))
        print('File:{}'.format(file_path))
        print('Start Index:{}'.format(start_index))
        try:
            for doc_indx, bert_embedding in tqdm(enumerate(embed_with_bert(documents_as_tokens[start_index:], start_index, file_names, args.window_length), start_index)):
                last_index = doc_indx # 5000
                if last_index < partition_number: # 5000 < 5000 True
                    if  args.ind_layers is not None:
                        token_embeddings = np.array(
                            [l for l_indx, l in enumerate(bert_embedding, 1) if  l_indx * -1 in ind_layers])
                        token_embeddings = np.average(token_embeddings, axis=0)
                    else:
                        token_embeddings = np.concatenate(
                            [l for l_indx, l in enumerate(bert_embedding, 1) if l_indx * -1 in conc_layers], axis=1)

                    print('generate_bert_embeddings')
                    print('*' * 20)
                    print('doc_indx:{}'.format(doc_indx))
                    print('len of tokens:{}'.format(len(documents_as_tokens[doc_indx])))
                    print('token embeddings:{}'.format(token_embeddings.shape))
                    print('tokens: {}'.format(documents_as_tokens[doc_indx]))
                    print('*' * 100)
                    print('*' * 100)

                    if args.is_inject_idf:
                        injected_idf_embeddings = []
                        for token in documents_as_tokens[doc_indx]:
                            injected_idf_embeddings.append(token2idfweight[token])
                        injected_idf_embeddings = np.asarray(injected_idf_embeddings).reshape(-1,1)
                        token_embeddings = np.multiply(token_embeddings, injected_idf_embeddings)
                    if args.is_averaged_token:
                        token_embeddings = np.mean(token_embeddings, axis=0)
                    document_embeddings.append(token_embeddings)
                else:
                    finalize_embeddings(document_embeddings, file_path, last_index)
                    document_embeddings = []
                    break
            if len(document_embeddings) != 0:
                finalize_embeddings(document_embeddings, file_path, last_index)
        except Exception as ex:
            print('last_index:{}'.format(last_index))
            raise Exception(ex)
    return name_prefix
def individually_generate_bert_embeddings(documents_as_tokens, path, args, conc_layers, ind_layers, partition_size, last_index, file_name_extension, token2idfweight, file_names):

    name_prefix = 'individually_' + file_name_extension + '_'
    try:
        start_index = last_index
        for doc_indx, bert_embedding in tqdm(enumerate(embed_with_bert(documents_as_tokens[start_index:], start_index, file_names, args.window_length), start_index)):
            file_path = os.path.join(path,
                                     name_prefix + str(doc_indx) + '.hdf5')
            last_index = doc_indx # 5000
            if  args.ind_layers is not None:
                token_embeddings = np.array(
                    [l for l_indx, l in enumerate(bert_embedding, 1) if  l_indx * -1 in ind_layers])
                token_embeddings = np.average(token_embeddings, axis=0)
            else:
                token_embeddings = np.concatenate(
                    [l for l_indx, l in enumerate(bert_embedding, 1) if l_indx * -1 in conc_layers], axis=1)

            print('generate_bert_embeddings')
            print('*' * 20)
            print('doc_indx:{}'.format(doc_indx))
            print('len of tokens:{}'.format(len(documents_as_tokens[doc_indx])))
            print('token embeddings:{}'.format(token_embeddings.shape))
            print('tokens: {}'.format(documents_as_tokens[doc_indx]))
            print('*' * 100)
            print('*' * 100)

            if args.is_inject_idf:
                injected_idf_embeddings = []
                for token in documents_as_tokens[doc_indx]:
                    injected_idf_embeddings.append(token2idfweight[token])
                injected_idf_embeddings = np.asarray(injected_idf_embeddings).reshape(-1,1)
                token_embeddings = np.multiply(token_embeddings, injected_idf_embeddings)
            token_embeddings = np.expand_dims(token_embeddings, axis=0)
            token_embeddings = pad(token_embeddings, args.max_tokens)
            UTIL.dump_embeddings(token_embeddings, file_path)
    except Exception as ex:
        print('last_index:{}'.format(last_index))
        raise Exception(ex)
    return name_prefix
def pad(x_matrix, max_tokens):
    new_x_matrix = None
    if max_tokens == -1:
        return x_matrix
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)
        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            temp_arr = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)
        else:
            temp_arr = np.delete(x_matrix[sentenceIdx], np.s_[max_tokens:], 0)
        if new_x_matrix is None:
            new_x_matrix = temp_arr
        else:
            new_x_matrix = np.vstack((new_x_matrix, temp_arr))
    #matrix = np.array(x_matrix, dtype=np.float32)
    return new_x_matrix


def embed_with_tfidf(documents_as_tokens):
    non_tokenized_documents = [" ".join(context) for context in documents_as_tokens]
    tfidf_matrix = np.array(tfidf_transformer.transform(non_tokenized_documents).todense())
    for doc_indx, doc in enumerate(non_tokenized_documents):
        token_embeddings = tfidf_matrix[doc_indx]
        print('*' * 100)
        print('*' * 100)
        print('embed_with_tfidf')
        print('*' * 20)
        print('doc_indx:{}'.format(doc_indx))
        print('token embeddings:{}'.format(token_embeddings.shape))
        print('=' * 20)
        yield np.expand_dims(np.expand_dims(token_embeddings,axis=0), axis=0)

def embed_with_bm25(documents_as_tokens):
    print('bm-25 is going to be calculated')
    for doc_indx, doc in enumerate(documents_as_tokens):
        token_embeddings = np.array(bm25_embeddings[bm25_index])
        print('*' * 100)
        print('*' * 100)
        print('embed_with_glove')
        print('*' * 20)
        print('doc_indx:{}'.format(doc_indx))
        print('token embeddings:{}'.format(token_embeddings.shape))
        print('=' * 20)
        yield np.expand_dims(np.expand_dims(token_embeddings, axis=0), axis=0)

def generate_tfidf(non_tokenized_documents, spacy_verbose=None, max_features=None):
    print('TF-IDF is going to be calculated')
    tokenize = lambda doc: [token.text for token in UTIL.word_tokenize(doc, spacy_verbose)]
    tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenize, max_features=max_features)
    tfidf.fit(non_tokenized_documents)
    return tfidf

def retrieve_IDF_weights(non_tokenized_documents, spacy_verbose=None):
    tfidf = generate_tfidf(non_tokenized_documents, spacy_verbose)
    max_idf = max(tfidf.idf_)
    token2idfweight = defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    return token2idfweight

def print_header(embedding_type, dataset_path):
    print('*' * 20)
    print('*' * 20)
    print('EMBEDDING TYPE:{}'.format(embedding_type))
    print('DATASET :{}'.format(dataset_path))
    print('*' * 20)
    print('*' * 20)

def load_fasttext_embedding(vocab, fasttext_file):
    en_model = KeyedVectors.load_word2vec_format(fasttext_file)
    embeddings = {}
    for word in vocab:
        if word not in embeddings:
            vector = [float(x) for x in en_model[word]]
        else:
            alpha = 0.5 * (2.0 * np.random.random() - 1.0)
            vector = (2.0 * np.random.random_sample([en_model.vector_size]) - 1.0) * alpha
        embeddings[word] = np.array(vector)
    return embeddings
def load_glove_embedding(vocab, w2v_file):
    """
    Pros:
        Save the oov words in oov.p for further analysis.
    Refs:
        class Vectors, https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    Args:
        vocab: dict,
        w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
    Returns:
        vectors
    """

    embeddings = {}  # (n_words, n_dim)

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    vectors, dim = array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(w2v_file, encoding="utf8") as f:
            lines = [line for line in f]
    # If there are malformed lines, read in binary mode
    # and manually decode each word from utf-8
    except:
        print("Could not read {} as UTF8 file, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(w2v_file))
        with open(w2v_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    print("Loading vectors from {}".format(w2v_file))

    for line in tqdm(lines):
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
            # init the embeddings
            #embeddings = np.random.uniform(-0.25, 0.25, (n_words, dim))

        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')

            except:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word in vocab:
            if word not in embeddings:
                # vectors = np.asarray(values[1:], dtype='float32')
                vector = [float(x) for x in entries]
            else:
                alpha = 0.5 * (2.0 * np.random.random() - 1.0)
                vector = (2.0 * np.random.random_sample([dim]) - 1.0) * alpha
            embeddings[word] = np.array(vector)

    # the words not in the embedding voc:
    for word in vocab:
        if word not in embeddings:
            alpha = 0.5 * (2.0 * np.random.random() - 1.0)
            vector = (2.0 * np.random.random_sample([dim]) - 1.0) * alpha
            embeddings[word] = np.array(vector)
    print('Total generated embeddings for this corpus is {}'.format(len(embeddings)))
    print('Total words in voc is {}'.format(len(vocab)))
    return embeddings

def main(args):
    ################ CONFIGURATIONS #################

    if args.dataset_path is None:
        raise Exception('A dataset path must be specified !!!')
    else:
        dataset_path = os.path.join(args.data_path, args.dataset_path)
        if args.is_read_contents_from_squad_format:
            tokenized_sources, tokenized_destinations, sources_nontokenized, destinations_nontokenized \
                = UTIL.prepare_squad_objects(squad_file = dataset_path,
                                             dataset_type = args.dataset_path,
                                             max_tokens=args.max_tokens,
                                             spacy_verbose=args.token_verbose)
        else:
            raise Exception('Reading from folders is not supported yet.')
            ##TODO: Reading from folders will be handling

        # os.path.isfile("bob.txt")
        # os.path.isdir("bob")

    ind_layers = None
    conc_layers = None
    # test_size = None
    # if args.test_size is not None:
    #     test_size = [int(x) for x in args.test_size.split(",")]

    if args.ind_layers is not None:
        ind_layers = [int(x) for x in args.ind_layers.split(",")]
        file_name_extension = 'ind_layers_' + args.ind_layers + '_token_verbose_' + str(args.token_verbose)
    else:
        conc_layers = [int(x) for x in args.conc_layers.split(",")]
        file_name_extension = 'conc_layers_' + args.conc_layers +'_token_verbose_' + str(args.token_verbose)

    if args.ind_layers is None and args.conc_layers is None:
        raise Exception('There must be some layer configurations !!!')
    if args.ind_layers is not None and args.conc_layers is not None:
        raise Exception('There must only one layer configuration !!!')

    _embedding_type = args.embedding_type.lower()
    _embedding_path = os.path.join(args.data_path, '#embedding_type#')
    _embedding_path = UTIL.create_dir(_embedding_path.replace("#embedding_type#", _embedding_type))
    source_folder_path = UTIL.create_dir(os.path.join(_embedding_path, 'source', 'injected_idf' if args.is_inject_idf else 'non_idf'))
    destination_folder_path = UTIL.create_dir(os.path.join(_embedding_path, 'destination', 'injected_idf' if args.is_inject_idf else 'non_idf'))
    print_header(args.embedding_type, args.dataset_path)
    token2idfweight = None
    if args.embedding_type in ['elmo', 'glove', 'tfidf','bm25', 'fasttext']:
        if args.embedding_type == 'elmo':
            elmo = ElmoEmbedder(cuda_device=0)
            embedding_function = elmo.embed_sentences
        elif args.embedding_type == 'glove':
            global glove_embeddings
            extracted_data_path = os.path.join(args.data_path, args.embedding_type,
                                                       'source_destination_glove_embeddings{}.pkl'.format(
                                                           '_token_verbose_' + str(args.token_verbose)))
            if not os.path.exists(extracted_data_path):
                voc = UTIL.create_vocabulary(tokenized_sources + tokenized_destinations)
                glove_embeddings = load_glove_embedding(voc, os.path.join(args.data_path,
                                                                          args.pre_generated_embeddings_path))


                print('Glove embeddings are generated')
                UTIL.save_as_pickle(glove_embeddings, extracted_data_path)
                print('Glove embeddings are dumped')
            else:
                glove_embeddings = UTIL.load_from_pickle(extracted_data_path)
                print('Glove embeddings are loaded')
            embedding_function = embed_with_glove
        elif args.embedding_type == 'tfidf':
            global tfidf_transformer
            train_dataset_path = os.path.join(args.data_path, args.pre_generated_embeddings_path)
            train_tokenized_sources, train_tokenized_destinations, train_sources_nontokenized, train_destinations_nontokenized \
                = UTIL.prepare_squad_objects(squad_file=train_dataset_path,
                                             dataset_type=args.dataset_path,
                                             max_tokens=args.max_tokens,
                                             spacy_verbose=args.token_verbose)

            tfidf_transformer = generate_tfidf(train_sources_nontokenized + train_destinations_nontokenized, args.token_verbose, 1024)
            # tfidf_transformer = generate_tfidf(sources_nontokenized +destinations_nontokenized,
            #                                    args.token_verbose, 1024)
            embedding_function = embed_with_tfidf
        elif args.embedding_type == 'bm25':
            global bm25_embeddings
            extracted_data_path = os.path.join(args.data_path, args.embedding_type,
                                               'source_destination_fasttext_embeddings{}.pkl'.format(
                                                   '_token_verbose_' + str(args.token_verbose)))
            if not os.path.exists(extracted_data_path):
                # train_dataset_path = os.path.join(args.data_path, args.pre_generated_embeddings_path)
                # train_tokenized_sources, train_tokenized_destinations, train_sources_nontokenized, train_destinations_nontokenized \
                #     = UTIL.prepare_squad_objects(squad_file=train_dataset_path,
                #                                  dataset_type=args.dataset_path,
                #                                  max_tokens=args.max_tokens,
                #                                  spacy_verbose=args.token_verbose)
                #
                # bm25_embeddings = get_bm25_weights(train_sources_nontokenized + train_destinations_nontokenized,n_jobs=-1)

                bm25_embeddings = get_bm25_weights(sources_nontokenized + destinations_nontokenized, n_jobs=-1)

                print('FastText embeddings are generated')
                UTIL.save_as_pickle(bm25_embeddings, extracted_data_path)
                print('FastText embeddings are dumped')
            else:
                bm25_embeddings = UTIL.load_from_pickle(extracted_data_path)
                print('FastText embeddings are loaded')

            embedding_function = embed_with_bm25
        elif args.embedding_type =='fasttext':
            global glove_embeddings
            extracted_data_path = os.path.join(args.data_path, args.embedding_type,
                                               'source_destination_fasttext_embeddings{}.pkl'.format(
                                                   '_token_verbose_' + str(args.token_verbose)))
            if not os.path.exists(extracted_data_path):
                voc = UTIL.create_vocabulary(tokenized_sources + tokenized_destinations)
                # fasttext_embeddings = load_fasttext_embedding(voc, os.path.join(args.data_path,
                #                                                           args.pre_generated_embeddings_path))
                glove_embeddings = load_glove_embedding(voc, os.path.join(args.data_path,
                                                                          args.pre_generated_embeddings_path))
                print('Glove embeddings are generated')
                UTIL.save_as_pickle(glove_embeddings, extracted_data_path)
                print('Glove embeddings are dumped')
            else:
                glove_embeddings = UTIL.load_from_pickle(extracted_data_path)
                print('Glove embeddings are loaded')
            embedding_function = embed_with_glove
        if args.embedding_type not in ['tfidf', 'bm25'] and args.is_inject_idf:
            token2idfweight = retrieve_IDF_weights(sources_nontokenized + destinations_nontokenized, args.token_verbose)
        if args.document_verbose == 1 or args.document_verbose == 3:
            print('SOURCE: Starting to embeddings generations')
            if args.is_averaged_token:
                source_folder_path = UTIL.create_dir(os.path.join(source_folder_path, 'paritionally'))
                source_name_prefix = partitionally_generate_generic_embeddings(tokenized_sources, source_folder_path, args, conc_layers, ind_layers, args.document_source_partition_size, args.document_source_index, file_name_extension, token2idfweight, embedding_function)
            else:
                source_folder_path = UTIL.create_dir(os.path.join(source_folder_path, 'individually'))
                source_name_prefix = individually_generate_generic_embeddings(tokenized_sources, source_folder_path, args, conc_layers,
                                                       ind_layers, args.document_source_partition_size,
                                                       args.document_source_index, file_name_extension, token2idfweight, embedding_function)
            print('SOURCE: Ending to embeddings generations')
        if args.document_verbose == 2 or args.document_verbose == 3:
            print('*' * 15)
            print('DESTINATION: Starting to embeddings generations')
            if args.is_averaged_token:
                destination_folder_path =  UTIL.create_dir(os.path.join(destination_folder_path, 'paritionally'))
                destination_name_prefix = partitionally_generate_generic_embeddings(tokenized_destinations, destination_folder_path, args, conc_layers, ind_layers,
                                         args.document_destination_partition_size, args.document_destination_index, file_name_extension,
                                         token2idfweight, embedding_function)
            else:
                destination_folder_path = UTIL.create_dir(os.path.join(destination_folder_path, 'individually'))
                destination_name_prefix = individually_generate_generic_embeddings(tokenized_destinations, destination_folder_path, args,
                                                       conc_layers, ind_layers,
                                                       args.document_destination_partition_size,
                                                       args.document_destination_index, file_name_extension,
                                                       token2idfweight, embedding_function)
            print('DESTINATION: Ending to embeddings generations')
    elif args.embedding_type == 'bert':
        if args.pre_generated_embeddings_path is None:
            raise Exception('There must a valid path for the source or source/destionation pre embeddings !!!')

        source_pre_generated_bert_embeddings_file_names = get_file_names(os.path.join(args.data_path, args.pre_generated_embeddings_path, 'source'), '_',
                                    '.json')
        destination_pre_generated_bert_embeddings_file_names = get_file_names(
            os.path.join(args.data_path, args.pre_generated_embeddings_path, 'destination'), '_',
            '.json')
        if args.is_inject_idf:
            # In order to get IDF weights, we have to calculate IDF weights according to the tokens of the bert.
            # so we have to grap all tokens for all documents and calculate their idf then do the rest.
            bert_new_source_tokens_file=os.path.join(args.data_path, args.embedding_type, 'bert_new_source{}.pkl'.format('_token_verbose_' + str(args.token_verbose)))
            bert_new_destionation_tokens_file = os.path.join(args.data_path, args.embedding_type,
                                                       'bert_new_destination{}.pkl'.format('_token_verbose_' + str(args.token_verbose)))
            if not os.path.exists(bert_new_source_tokens_file):
                tokenized_sources = convert_bert_tokens(tokenized_sources, source_pre_generated_bert_embeddings_file_names, args.window_length)
                print('SOURCE: New tokens are generated')
                UTIL.save_as_pickle(tokenized_sources, bert_new_source_tokens_file)
                print('SOURCE: New tokens are dumped')
            else:
                tokenized_sources = UTIL.load_from_pickle(bert_new_source_tokens_file)
                print('SOURCE: New tokens are loaded')

            if not os.path.exists(bert_new_destionation_tokens_file):
                tokenized_destinations = convert_bert_tokens(tokenized_destinations, destination_pre_generated_bert_embeddings_file_names, args.window_length)
                print('DESTINATION: New tokens are generated')
                UTIL.save_as_pickle(tokenized_destinations, bert_new_destionation_tokens_file)
                print('DESTINATION: New tokens are dumped')
            else:
                tokenized_destinations = UTIL.load_from_pickle(bert_new_destionation_tokens_file)
                print('DESTINATION: New tokens are loaded')

            sources_nontokenized = [" ".join(context) for context in tokenized_sources]
            destinations_nontokenized = [" ".join(context) for context in tokenized_destinations]
            token2idfweight = retrieve_IDF_weights(sources_nontokenized + destinations_nontokenized)

        if args.document_verbose == 1 or args.document_verbose == 3:
            print('SOURCE: Starting to embeddings generations')
            if args.is_averaged_token:
                source_folder_path = UTIL.create_dir(os.path.join(source_folder_path, 'paritionally'))
                source_name_prefix = partitionally_generate_bert_embeddings(tokenized_sources, source_folder_path, args, conc_layers, ind_layers, args.document_source_partition_size, args.document_source_index, file_name_extension, token2idfweight, source_pre_generated_bert_embeddings_file_names)
            else:
                source_folder_path = UTIL.create_dir(os.path.join(source_folder_path, 'individually'))
                source_name_prefix = individually_generate_bert_embeddings(tokenized_sources, source_folder_path, args, conc_layers,
                                                       ind_layers, args.document_source_partition_size,
                                                       args.document_source_index, file_name_extension, token2idfweight,
                                                       source_pre_generated_bert_embeddings_file_names)
            print('SOURCE: Ending to embeddings generations')
        if args.document_verbose == 2 or args.document_verbose == 3:
            print('*' * 15)
            print('SOURCE: Starting to embeddings generations')
            if args.is_averaged_token:
                destination_folder_path = UTIL.create_dir(os.path.join(destination_folder_path, 'paritionally'))
                destination_name_prefix = partitionally_generate_bert_embeddings(tokenized_destinations, destination_folder_path, args, conc_layers, ind_layers,
                                         args.document_destination_partition_size, args.document_destination_index, file_name_extension,
                                         token2idfweight, destination_pre_generated_bert_embeddings_file_names)
            else:
                destination_folder_path = UTIL.create_dir(os.path.join(destination_folder_path, 'individually'))
                destination_name_prefix = individually_generate_bert_embeddings(tokenized_destinations, destination_folder_path, args,
                                                       conc_layers, ind_layers,
                                                       args.document_destination_partition_size,
                                                       args.document_destination_index, file_name_extension,
                                                       token2idfweight,
                                                       destination_pre_generated_bert_embeddings_file_names)
            print('SOURCE: Ending to embeddings generations')

    else:
        raise Exception('There is no such embedding or is not supported yet.')

    if args.is_stack_all:
        if args.document_verbose == 1 or args.document_verbose == 3:
            print('SOURCES: Starting to stack embeddings')
            stack_partitioned_embeddings(source_folder_path, file_name_extension, source_name_prefix, args.document_source_partition_size, args.is_averaged_token)
            print('SOURCES: Ending to stack embeddings')
        if args.document_verbose == 2 or args.document_verbose == 3:
            print('*' * 15)
            print('DESTINATION: Starting to stack embeddings')
            stack_partitioned_embeddings(destination_folder_path, file_name_extension, destination_name_prefix, args.document_destination_partition_size, args.is_averaged_token)
            print('DESTINATION: Ending to stack embeddings')

    # ################ CONFIGURATIONS #################


if __name__ == '__main__':
    """
    sample executions: 

    """
    args = get_parser().parse_args()
    assert args.data_path is not None, "No folder path found at {}".format(args.data_path)
    # assert args.to_file_name is not None, "No 'to_file_name' found {}".format(args.to_file_name)
    main(args)


