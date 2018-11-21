import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import re
import sys
import numpy as np
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
#import squad.wordpiece_tokenization as wordpiece
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', default=4, type=int, help="-1: no windows, else should be selected from the given range 1-512. it should be at least 1 lower than truncate_length")
    parser.add_argument('--data_path', help="path to the source files, this folder(dev/train) must contain questions/paragraphs folder for embeddings")
    parser.add_argument('--squad_formatted_file', help='squad formatted dataset')
    parser.add_argument('--conc_layers', default="-1,-2,-3,-4", help='whether to concatenate all specified layers or None')
    parser.add_argument('--ind_layer', default=None,
                        help='whether to create individual representations for specified layer or None')
    parser.add_argument('--is_inject_idf', default=False, type=bool, help="whether inject idf or not to the weights")
    parser.add_argument('--is_parititioned', default=True, type=bool,
                        help="handle file read/write partially")
    parser.add_argument('--token_partition_size', default=50000, type=int,
                        help="size of partition to handle tokens")
    parser.add_argument('--document_partition_size', default=5000, type=int,
                        help="size of partition to handle documents")
    parser.add_argument('--test_size', default=None,
                        help="question, paragraph sizes")
    return parser
def get_file_names(path, file_name_splitter, bert_extension):
    bert_embeddings_file_names = []
    for name in [name for name in os.listdir(path)
                  if name.endswith(bert_extension)]:
        _names = name.rpartition('.')[0].split(file_name_splitter)[3:6]
        _names.remove('to')
        item = [int(index) for index in _names] + [name]
        bert_embeddings_file_names.append(item)
    bert_embeddings_file_names.sort()
    return bert_embeddings_file_names

def find_file_name(index, file_names):
    for file_name in file_names:
        if file_name[0] <= index <= file_name[1]:
            return file_name[2], file_name[0] - index

def process_documents(partition, document_partition_size, checkpoint, tokenized_document_size, file_names, file_folder_path, ind_layer, conc_layers, all_tokens, contextualized_questions_with_token_file_path):
    embeddings = None
    start = partition
    end =  (partition + document_partition_size) if document_partition_size is not None else tokenized_document_size
    for indx in tqdm(range(start, end)):
        if tokenized_document_size > indx:
            bert_index = indx + 1
            file_name, remaining_index_to_pass_this_file = find_file_name(bert_index, file_names)
            if remaining_index_to_pass_this_file >= 0:
                jsons = UTIL.load_bert_jsons_from_single_file(os.path.join(file_folder_path, file_name))
                if indx > 0:
                    checkpoint = indx
            if checkpoint is not None:
                indx = indx - checkpoint
            new_token = []
            token_embeddings = None
            for line_index, json in UTIL.reversedEnumerate(jsons[indx]):
                # 0 and -1 token indexes belong to [CLS, SEP] we are ignoring them.
                json['features'].pop(0)
                json['features'].pop(-1)

                # filter out the non-contributional tokens from the list.
                features = [x for x in json['features'] if not x['token'].startswith("##")]
                for feature_index, feature in UTIL.reversedEnumerate(features):
                    if line_index > 0 and feature_index < args.window_length:
                        # print(feature['token'])
                        continue

                    if args.ind_layer is not None:
                        token_embedding = np.array(
                            [l['values'] for l in feature['layers'] if l['index'] == ind_layer])
                    else:
                        token_embedding = np.concatenate(
                            [l['values'] for l in feature['layers'] if l['index'] in conc_layers])

                    if token_embeddings is None:
                        token_embeddings = token_embedding
                    else:
                        token_embeddings = np.vstack((token_embeddings, token_embedding))

                    new_token.append(feature['token'])
            if len(new_token) != token_embeddings.shape[0]:
                print(30 * '*')
                print('********** Size of token embeddings {} has problem in {} checkpoint **********'.format(
                    indx, checkpoint))
                print(30 * '*')
            all_tokens.append(new_token)
            if embeddings is None:
                embeddings = token_embeddings
            else:
                embeddings = np.vstack((embeddings, token_embeddings))

    print('embeddings shape: {}'.format(embeddings.shape))
    UTIL.dump_embeddings(embeddings, contextualized_questions_with_token_file_path)
    print('embeddings are dumped')

def main(args):

    ################ CONFIGURATIONS #################
    squad_formatted_file = os.path.join(args.data_path, args.squad_formatted_file)
    bert_extension = ".json"
    file_name_splitter = '_'
    document_embeddings = None
    questions_folder_path = os.path.join(args.data_path, 'questions')
    paragraphs_folder_path = os.path.join(args.data_path, 'paragraphs')
    new_question_tokens_path = os.path.join(args.data_path, 'questions_tokens.pkl')
    new_paragraph_tokens_path = os.path.join(args.data_path, 'paragraphs_tokens.pkl')
    calculated_token_embeddings_file_path= os.path.join(args.data_path, 'contextualized_document_embeddings_with_token_##_@@.hdf5')
    vocab_path = os.path.join(args.data_path, 'wordpiece_vocab.txt')
    ind_layer = None
    conc_layers = None
    test_size= None
    if args.test_size is not None:
        test_size = [int(x) for x in args.test_size.split(",")]
    if args.ind_layer is not None:
        ind_layer= int(args.ind_layer)
        contextualized_questions_with_token_file_path = os.path.join(args.data_path, "contextualized_questions_embeddings_with_tokens_{}_layers_@@.hdf5".format(args.ind_layer))
        contextualized_paragraphs_with_token_file_path = os.path.join(args.data_path,
                                                                     "contextualized_paragraphs_embeddings_with_tokens_{}_layers.hdf5_@@".format(
                                                                         args.ind_layer))
        contextualized_document_embeddings_with_token_path = os.path.join(args.data_path,
                                                                     "contextualized_document_embeddings_with_token_{}_layers.hdf5".format(
                                                                         args.ind_layer))
        final_questions_file_path = os.path.join(args.data_path,
                                                                          "question_document_embeddings_{}_layers_@@.hdf5".format(
                                                                              args.ind_layer))
        final_paragraphs_file_path = os.path.join(args.data_path,
                                                 "paragraph_document_embeddings_{}_layers_@@.hdf5".format(
                                                     args.ind_layer))
    else:
        conc_layers = [int(x) for x in args.conc_layers.split(",")]
        contextualized_questions_with_token_file_path = os.path.join(args.data_path,
                                                                     "contextualized_questions_embeddings_with_tokens_{}_layers_@@.hdf5".format(
                                                                         conc_layers))
        contextualized_paragraphs_with_token_file_path = os.path.join(args.data_path,
                                                                     "contextualized_paragraphs_embeddings_with_tokens_{}_layers_@@.hdf5".format(
                                                                         conc_layers))
        contextualized_document_embeddings_with_token_path = os.path.join(args.data_path,
                                                                          "contextualized_document_embeddings_with_token_{}_layers.hdf5".format(
                                                                              conc_layers))
        final_questions_file_path = os.path.join(args.data_path,
                                                 "question_document_embeddings_{}_layers_@@.hdf5".format(
                                                     conc_layers))
        final_paragraphs_file_path = os.path.join(args.data_path,
                                                  "paragraph_document_embeddings_{}_layers_@@.hdf5".format(
                                                      conc_layers))
    if args.ind_layer is None and args.conc_layers is None:
        raise Exception('There must be some layer configurations !!!')
    if args.ind_layer is not None and args.conc_layers is not None:
        raise Exception('There must only one layer configuration !!!')
    # ################ CONFIGURATIONS #################


    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    START: PARSING FILE
    ******************************************************************************************************************
    ******************************************************************************************************************
    """
    tokenized_questions, tokenized_paragraphs, questions_nontokenized, paragraphs_nontokenized = UTIL.prepare_squad_objects(squad_formatted_file, args.squad_formatted_file)
    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    END: PARSING FILE
    ******************************************************************************************************************
    ******************************************************************************************************************
    """

    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    START: LOAD EMBEDINGS
    ******************************************************************************************************************
    ******************************************************************************************************************
    """
    new_question_tokens = []
    if os.path.exists(contextualized_questions_with_token_file_path):
        question_embeddings = UTIL.load_embeddings(contextualized_questions_with_token_file_path.replace('@@', ''))
        new_question_tokens = UTIL.load_from_pickle(new_question_tokens_path)
    else:
        file_names = get_file_names(questions_folder_path, file_name_splitter, bert_extension)
        tokenized_questions_size = test_size[0] if test_size is not None else len(tokenized_questions)
        checkpoint = None
        if args.is_parititioned is True:
            partition_counter = 0
            print("Partition {} is running for writing questions".format(partition_counter))
            for _p_counter in tqdm(range(0, tokenized_questions_size, args.document_partition_size)):
                process_documents(_p_counter, args.document_partition_size, checkpoint, tokenized_questions_size, file_names, questions_folder_path, ind_layer, conc_layers, new_question_tokens,contextualized_questions_with_token_file_path.replace('@@', str(partition_counter)) )
                partition_counter += 1

            question_embeddings = None
            print("Partition {} is running for reading questions".format(partition_counter))
            for _p_counter in tqdm(range(0, partition_counter)):
                temp_question_embeddings = UTIL.load_embeddings(
                    contextualized_questions_with_token_file_path.replace("@@", str(_p_counter)))
                if question_embeddings is None:
                    question_embeddings = temp_question_embeddings
                else:
                    question_embeddings = np.vstack((question_embeddings, temp_question_embeddings))
            print('MAIN embeddings shape: {}'.format(question_embeddings.shape))
            UTIL.dump_embeddings(question_embeddings, contextualized_questions_with_token_file_path.replace('@@', ''))
            print('MAIN embeddings are dumped')
        else:
            print("It is running for writing questions")
            process_documents(0, None, checkpoint, tokenized_questions_size, file_names, questions_folder_path,
                              ind_layer, conc_layers, new_question_tokens,
                              contextualized_questions_with_token_file_path.replace('@@', ''))
        UTIL.save_as_pickle(new_question_tokens, new_question_tokens_path)

    ## ***************************************************************************************************************
    ## ***************************************************************************************************************
    ## ***************************************************************************************************************
    new_paragraph_tokens = []
    if os.path.exists(contextualized_paragraphs_with_token_file_path):
        paragraph_embeddings = UTIL.load_embeddings(contextualized_paragraphs_with_token_file_path)
        new_paragraph_tokens = UTIL.load_from_pickle(new_paragraph_tokens_path)
    else:
        file_names = get_file_names(paragraphs_folder_path, file_name_splitter, bert_extension)
        tokenized_paragraphs_size = test_size[1] if test_size is not None else len(tokenized_paragraphs)
        checkpoint=None
        if args.is_parititioned is True:
            partition_counter = 0
            print("Partition {} is running for writing paragraphs".format(partition_counter))
            for _p_counter in tqdm(range(0, tokenized_paragraphs_size, args.document_partition_size)):
                process_documents(_p_counter, args.document_partition_size, checkpoint, tokenized_paragraphs_size, file_names, paragraphs_folder_path,
                                  ind_layer, conc_layers, new_question_tokens,
                                  contextualized_paragraphs_with_token_file_path.replace('@@', str(partition_counter)))
                partition_counter += 1
            paragraph_embeddings = None
            print("Partition {} is running for reading paragraphs".format(partition_counter))
            for _p_counter in tqdm(range(0, partition_counter)):
                temp_paragraph_embeddings = UTIL.load_embeddings(
                    contextualized_paragraphs_with_token_file_path.replace("@@", str(_p_counter)))
                if paragraph_embeddings is None:
                    paragraph_embeddings = temp_paragraph_embeddings
                else:
                    paragraph_embeddings = np.vstack((paragraph_embeddings, temp_question_embeddings))
            print('MAIN embeddings shape: {}'.format(question_embeddings.shape))
            UTIL.dump_embeddings(paragraph_embeddings, contextualized_paragraphs_with_token_file_path.replace('@@', ''))
            print('MAIN embeddings are dumped')

        else:
            print("It is running for writing paragraphs")
            process_documents(0, None, checkpoint, tokenized_paragraphs_size, file_names, questions_folder_path,
                              ind_layer, conc_layers, new_question_tokens,
                              contextualized_paragraphs_with_token_file_path.replace('@@', ''))
        UTIL.save_as_pickle(new_paragraph_tokens, new_paragraph_tokens_path)

    if os.path.exists(contextualized_document_embeddings_with_token_path):
        if args.is_parititioned is not True:
            document_embeddings = UTIL.load_embeddings(contextualized_document_embeddings_with_token_path)
    else:
        document_embeddings = np.vstack((question_embeddings, paragraph_embeddings ))
        UTIL.dump_embeddings(document_embeddings, contextualized_document_embeddings_with_token_path)
    del question_embeddings
    del paragraph_embeddings
    print('All Documents are dumped')

    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    END: LOAD EMBEDINGS
    ******************************************************************************************************************
    ******************************************************************************************************************
    """

    document_embedding_guideline, corpus_as_tokens = UTIL.generate_document_embedding_guideline(new_question_tokens,
                                                                                                new_paragraph_tokens)


    paragraphs_nontokenized = [" ".join(context) for context in new_paragraph_tokens]
    questions_nontokenized = [" ".join(context) for context in new_question_tokens]
    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    START: IDF
    ******************************************************************************************************************
    ******************************************************************************************************************
    """
    if args.is_inject_idf:
        print('IDF is going to be calculated')
        # vocab = []
        # for sentence in new_question_tokens + new_paragraph_tokens:
        #     for word in sentence:
        #         vocab.append(word)
        # vocab = set(vocab)
        # UTIL.dump_vocab(vocab_path, vocab)
        #tokenize = wordpiece.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
        nlp = spacy.blank("en")
        tokenize = lambda doc: [token.text for token in nlp(doc)]
        start = datetime.datetime.now()
        token2idfweight, idf_vec = UTIL.transform_to_idf_weigths(new_question_tokens,
                                                                 new_paragraph_tokens,
                                                                 tokenize,
                                                                 questions_nontokenized,
                                                                 paragraphs_nontokenized)
        if args.is_parititioned is True:
            with h5py.File(contextualized_document_embeddings_with_token_path, 'r') as fin:
                partition_counter=0
                for partition in range(0,idf_vec.shape[0], args.token_partition_size):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args.token_partition_size, :]
                    temp_idf_vec = idf_vec[partition:partition + args.token_partition_size, :].reshape(-1,1)
                    #temp_doc_embeddings = temp_doc_embeddings[:,0,:]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    temp_weighted_token_embeddings = np.multiply(temp_idf_vec, temp_doc_embeddings)
                    UTIL.dump_embeddings(temp_weighted_token_embeddings, calculated_token_embeddings_file_path.replace('@@', str(partition_counter)).replace('##', 'idf'))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition, partition + args.token_partition_size))
        else:
            idf_vec = idf_vec.reshape(-1, 1)
            weighted_token_embeddings = np.multiply(idf_vec, document_embeddings)
        del idf_vec
        del token2idfweight
        end = datetime.datetime.now()
        print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
    else:
        print('IDF is skipped')
        _type = 'only'
        if args.is_parititioned is True:
            with h5py.File(contextualized_document_embeddings_with_token_path, 'r') as fin:
                partition_counter = 0
                for partition in range(0, len(corpus_as_tokens), args.token_partition_size):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args.token_partition_size, :]
                    #temp_doc_embeddings = temp_doc_embeddings[:, 0, :]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    UTIL.dump_embeddings(temp_doc_embeddings, calculated_token_embeddings_file_path.replace('@@', str(partition_counter)).replace('##', ''))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition,
                                                                                          partition + args.token_partition_size))
        else:
            weighted_token_embeddings = document_embeddings
    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    END: LOAD IDF
    ******************************************************************************************************************
    ******************************************************************************************************************
    """

    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    START: WEIGHTED ARE GETTING APPLIED TO TOKEN EMBEDDINGS
    ******************************************************************************************************************
    ******************************************************************************************************************
    """
    del document_embeddings


    #LOAD PARTIAL FILES AFTER CLEANING THE DOCUMENT EMBEDDINGS.
    if args.is_parititioned is True:
        weighted_token_embeddings = None
        for partition in range(1, partition_counter+1):
            temp_weighted_token_embeddings = UTIL.load_embeddings(calculated_token_embeddings_file_path.replace('@@', str(partition)).replace('##', 'idf' if args.is_inject_idf else ''))
            if weighted_token_embeddings is None:
                weighted_token_embeddings = temp_weighted_token_embeddings
            else:
                weighted_token_embeddings = np.vstack((weighted_token_embeddings, temp_weighted_token_embeddings))
            print("Partition {} is loaded".format(partition))


    WM = None #np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
    questions_embeddings, paragraphs_embeddings = UTIL.token_to_document_embeddings(new_question_tokens,
                                                                                new_paragraph_tokens,
                                                                                weighted_token_embeddings,
                                                                                document_embedding_guideline,
                                                                                WM
                                                                                )

    if args.is_inject_idf:
        questions_elmo_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[1]))
        UTIL.dump_embeddings(questions_elmo_embeddings, final_questions_file_path.replace('@@', 'with_idf'))
        paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                                   (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
        UTIL.dump_embeddings(paragraphs_elmo_embeddings, final_paragraphs_file_path.replace('@@', 'with_idf'))
    else:
        questions_elmo_embeddings = np.reshape(questions_embeddings,
                                               (questions_embeddings.shape[0], questions_embeddings.shape[1]))
        UTIL.dump_embeddings(questions_elmo_embeddings, final_questions_file_path.replace('@@', ''))
        paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                                (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
        UTIL.dump_embeddings(paragraphs_elmo_embeddings, final_paragraphs_file_path.replace('@@', ''))

    print('Weighted are applied')
    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    END : WEIGHTED ARE GETTING APPLIED TO TOKEN EMBEDDINGS
    ******************************************************************************************************************
    ******************************************************************************************************************
    """

if __name__ == '__main__':
    """
    sample executions: 

    """
    args = get_parser().parse_args()
    assert args.data_path is not None, "No folder path found at {}".format(args.data_path)
    # assert args.to_file_name is not None, "No 'to_file_name' found {}".format(args.to_file_name)
    main(args)
