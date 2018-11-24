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
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', default=4, type=int, help="-1: no windows, else should be selected from the given range 1-512. it should be at least 1 lower than truncate_length")
    parser.add_argument('--data_path', help="as a master path to locate all files and older underneath")
    parser.add_argument('--is_read_contents_from_squad_format', type=UTIL.str2bool)
    parser.add_argument('--max_tokens', default=-1, type=int, help='Whether to limit the number of tokens per context')
    parser.add_argument('--dataset_path', type=str, help='whether squad formatted json file or a folder that has individual files')
    parser.add_argument('--conc_layers', default=None, help='whether to concatenate all specified layers or None')
    parser.add_argument('--ind_layers', default=None,
                        help='whether to create individual representations for specified layer or None, 0 for token layer')
    parser.add_argument('--is_inject_idf', default=True, type=UTIL.str2bool, help="whether inject idf or not to the weights")
    parser.add_argument('--is_averaged_token', default=True, type=UTIL.str2bool,
                        help="For FCN models, it must be set to True, for convolution kernels, it must be set to False, but the generated files will be very large")
    parser.add_argument('--is_stack_all_partitioned', default=True, type=UTIL.str2bool,
                        help="in order to stack all embeddings and make one file.")
    parser.add_argument('--document_source_partition_size', default=2000, type=int,
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
    parser.add_argument('--is_powerful_gpu', default=True, type=UTIL.str2bool,
                        help="whether it is a high gpu or not")
    return parser


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

def stack_partitioned_embeddings(path, file_name_extension):
    name_prefix = 'partitioned_' + file_name_extension + '_'
    names=[]
    for name in [name for name in os.listdir(path)
                 if name.startswith(name_prefix)]:
        _name = int(name.rpartition('_')[2].rpartition('.')[0])
        names.append(_name)
    names.sort()
    print('File names: {}'.format(names))
    embeddings = None
    for name in names:
        name_path = os.path.join(path, name_prefix + str(name) + '.hdf5')
        embedding = UTIL.load_embeddings(name_path)
        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = np.vstack((embeddings, embedding))

    UTIL.dump_embeddings(embeddings, os.path.join(path, 'all_embeddings_'+file_name_extension + '.hdf5'))
    print("Embeddings are getting dumped {}".format(embeddings.shape))

def generate_elmo_embeddings(elmo, documents_as_tokens, path, args, conc_layers, ind_layers, partition_size, last_index, file_name_extension, token2idfweight):
    document_embeddings = []

    # if args.ind_layer is not None:
    #     if ind_layer == 0:
    #         elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    #
    # else:
    for partition_counter in range(0, len(documents_as_tokens), partition_size):
        partition_number = partition_counter + partition_size
        file_path = os.path.join(path, 'partitioned_' +  file_name_extension + '_' + str(partition_counter) + '.hdf5')
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

        for doc_indx, elmo_embedding in tqdm(enumerate(elmo.embed_sentences(documents_as_tokens[start_index:]), start_index)):
            last_index = doc_indx # 5000
            if last_index < partition_number: # 5000 < 5000 True
                if  args.ind_layers is not None:
                    token_embeddings = np.array(
                        [l for l_indx, l in enumerate(reversed(elmo_embedding), 1) if  l_indx * -1 in ind_layers])
                    token_embeddings = np.average(token_embeddings, axis=0)
                else:
                    token_embeddings = np.concatenate(
                        [l for l_indx, l in enumerate(elmo_embedding, 1) if l_indx * -1 in conc_layers], axis=1)
                if args.is_averaged_token:
                    injected_idf_embeddings = []
                    if args.is_inject_idf:
                        for token in documents_as_tokens[doc_indx]:
                            injected_idf_embeddings.append(token2idfweight[token])
                        injected_idf_embeddings = np.asarray(injected_idf_embeddings).reshape(-1,1)
                        token_embeddings = np.multiply(token_embeddings, injected_idf_embeddings)
                    token_embeddings = np.mean(token_embeddings, axis=0)
                document_embeddings.append(token_embeddings)
            else:
                finalize_embeddings(document_embeddings, file_path, last_index)
                document_embeddings = []
                break
        if len(document_embeddings) != 0:
            finalize_embeddings(document_embeddings, file_path, last_index)

    # return embeddings, labels

def pad(x_matrix, max_tokens):
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)
        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            x_matrix[sentenceIdx] = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)

    matrix = np.array(x_matrix, dtype=np.float32)
    return matrix

def retrieve_IDF_weights(non_tokenized_documents):
    print('IDF is going to be calculated')
    nlp = spacy.blank("en")
    tokenize = lambda doc: [token.text for token in nlp(doc)]
    tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenize)
    tfidf.fit(non_tokenized_documents)
    max_idf = max(tfidf.idf_)
    token2idfweight = defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    return token2idfweight


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
                                             max_tokens=args.max_tokens)
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
        file_name_extension = 'ind_layers_' + args.ind_layers
    else:
        conc_layers = [int(x) for x in args.conc_layers.split(",")]
        file_name_extension = 'conc_layers_' + args.conc_layers

    if args.ind_layers is None and args.conc_layers is None:
        raise Exception('There must be some layer configurations !!!')
    if args.ind_layers is not None and args.conc_layers is not None:
        raise Exception('There must only one layer configuration !!!')

    _embedding_type = args.embedding_type.lower()
    _embedding_path = os.path.join(args.data_path, '#embedding_type#')
    _embedding_path = UTIL.create_dir(_embedding_path.replace("#embedding_type#", _embedding_type))
    source_folder_path = UTIL.create_dir(os.path.join(_embedding_path, 'source', 'injected_idf' if args.is_inject_idf else 'non_idf'))
    destination_folder_path = UTIL.create_dir(os.path.join(_embedding_path, 'destination', 'injected_idf' if args.is_inject_idf else 'non_idf'))
    if args.embedding_type == 'elmo':
        elmo = ElmoEmbedder(cuda_device=0)
        token2idfweight = None
        if args.is_inject_idf:
            token2idfweight = retrieve_IDF_weights(sources_nontokenized + destinations_nontokenized)
        print('SOURCES: Starting to embeddings generations')
        generate_elmo_embeddings(elmo,tokenized_sources, source_folder_path, args, conc_layers, ind_layers, args.document_source_partition_size, args.document_source_index, file_name_extension, token2idfweight)
        print('SOURCES: Ending to embeddings generations')
        if args.is_powerful_gpu:
            print('*' * 15)
            print('DESTINATION: Starting to embeddings generations')
            generate_elmo_embeddings(elmo, tokenized_destinations, destination_folder_path, args, conc_layers, ind_layers,
                                     args.document_destination_partition_size, args.document_destination_index, file_name_extension,
                                     token2idfweight)
            print('DESTINATION: Ending to embeddings generations')
    elif args.embedding_type == 'bert':
        #TODO: Bert
        pass
    elif args.embedding_type == 'glove':
        #TODO: Glove
        pass
    else:
        raise Exception('There is no such embedding or is not supported yet.')

    if args.is_stack_all_partitioned:
        print('SOURCES: Starting to stack embeddings')
        stack_partitioned_embeddings(source_folder_path, file_name_extension)
        print('SOURCES: Ending to stack embeddings')
        print('*' * 15)
        if args.is_powerful_gpu:
            print('DESTINATION: Starting to stack embeddings')
            stack_partitioned_embeddings(destination_folder_path, file_name_extension)
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


