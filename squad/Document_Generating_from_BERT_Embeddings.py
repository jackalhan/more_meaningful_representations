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
import squad.wordpiece_tokenization as wordpiece
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
    parser.add_argument('--is_parititioned', default=False, type=bool,
                        help="handle file read/write partially")
    parser.add_argument('--partition_size', default=100000,
                        help="size of partition to handle tokens")
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

def main(args):
    # ################ CONFIGURATIONS #################
    #
    # NEW_API_ELMO={"is_inject_idf":True,
    # "load_data_partially": False,
    # "partition_size": None,
    # "calculated_idf_token_embeddings_file": '{}_contextualized_document_embeddings_with_token_and_idf_@@.hdf5'.format(dataset_type),
    #       "root_path": "ELMO_CONTEXT_NEW_API_EMBEDDINGS",
    #       "embedding_paragraphs_path": "paragraphs",
    #       "embedding_paragraphs_file_pattern": "{}_paragraph_embedding_LSTM1_@@.hdf5".format(dataset_type),
    #       "contextualized_paragraphs_embeddings_with_token": '{}_contextualized_paragraphs_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
    #       "embedding_questions_path": "questions",
    #       "embedding_questions_file_pattern": "{}_question_embeddings_LSTM1_@@.hdf5".format(dataset_type),
    #       "contextualized_questions_embeddings_with_token": '{}_contextualized_questions_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
    #        "is_paragraphs_listed_after_questions":False,
    #       "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
    #       "change_shape": False,
    #       "weights_arguments": [1],
    #       'questions_file': '{}_question_document_embeddings_LSTM1_@@.hdf5'.format(dataset_type),
    #       'paragraphs_file': '{}_paragraph_document_embeddings_LSTM1_@@.hdf5'.format(dataset_type),
    #       'is_calculate_recalls': True,
    #       'recall_file_path': '{}_recalls_weights_LSTM1_@@_###.csv'.format(dataset_type)
    #       }
    #
    # OLD_API_ELMO={"is_inject_idf":True,
    #               "load_data_partially": True,
    #               "partition_size": 100000,
    #               "calculated_token_embeddings_file": '{}_contextualized_document_embeddings_with_token_##_@@.hdf5'.format(dataset_type),
    #       "root_path": "ELMO_CONTEXT_OLD_API_EMBEDDINGS",
    #       "embedding_paragraphs_path": None,
    #       "embedding_paragraphs_file_pattern": "{}_token_embeddings_old_api_doc_@@.hdf5".format(dataset_type),
    #       "contextualized_paragraphs_embeddings_with_token": '{}_contextualized_paragraphs_embeddings_with_token.hdf5'.format(dataset_type),
    #       "embedding_questions_path": None,
    #       "embedding_questions_file_pattern": "{}_token_embeddings_old_api_doc_@@.hdf5".format(dataset_type),
    #       "contextualized_questions_embeddings_with_token": '{}_contextualized_questions_embeddings_with_token.hdf5'.format(dataset_type),
    #       "is_paragraphs_listed_after_questions":True,
    #       "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token.hdf5'.format(dataset_type),
    #       "change_shape": False,
    #       "weights_arguments": [1, 0, 0], #icerde max index olarak kullanilabilr.
    #       'questions_file': '{}_question_document_embeddings_@@.hdf5'.format(dataset_type),
    #       'paragraphs_file': '{}_paragraph_document_embeddings_@@.hdf5'.format(dataset_type),
    #       'is_calculate_recalls': False,
    #       'recall_file_path': '{}_recalls_weights_@@_###.csv'.format(dataset_type)
    #       }

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
        contextualized_questions_with_token_file_path = os.path.join(args.data_path, "contextualized_questions_embeddings_with_tokens_{}_layers.hdf5".format(args.ind_layer))
        contextualized_paragraphs_with_token_file_path = os.path.join(args.data_path,
                                                                     "contextualized_paragraphs_embeddings_with_tokens_{}_layers.hdf5".format(
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
                                                                     "contextualized_questions_embeddings_with_tokens_{}_layers.hdf5".format(
                                                                         conc_layers))
        contextualized_paragraphs_with_token_file_path = os.path.join(args.data_path,
                                                                     "contextualized_paragraphs_embeddings_with_tokens_{}_layers.hdf5".format(
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
    question_embeddings = None
    new_question_tokens = []
    if os.path.exists(contextualized_questions_with_token_file_path):
        question_embeddings = UTIL.load_embeddings(contextualized_questions_with_token_file_path)
        new_question_tokens = UTIL.load_from_pickle(new_question_tokens_path)
    else:
        file_names = get_file_names(questions_folder_path, file_name_splitter, bert_extension)
        tokenized_questions_size = test_size[0] if test_size is not None else len(tokenized_questions)
        checkpoint = None
        for question_indx in range(tokenized_questions_size):
            question_bert_index = question_indx + 1
            file_name, remaining_index_to_pass_this_file = find_file_name(question_bert_index, file_names)
            if remaining_index_to_pass_this_file >= 0:
                jsons = UTIL.load_bert_jsons_from_single_file(os.path.join(questions_folder_path, file_name))
                if question_indx > 0:
                    checkpoint = question_indx
            if checkpoint is not None:
                question_indx = question_indx - checkpoint
            new_question_token = []
            for line_index, json in UTIL.reversedEnumerate(jsons[question_indx]):
                # 0 and -1 token indexes belong to [CLS, SEP] we are ignoring them.
                json['features'].pop(0)
                json['features'].pop(-1)

                # filter out the non-contributional tokens from the list.
                features = [x for x in json['features'] if not x['token'].startswith("##")]
                token_embeddings = None
                for feature_index, feature in UTIL.reversedEnumerate(features):
                    if line_index > 0 and feature_index <= args.window_length:
                        #print(feature['token'])
                        continue

                    if args.ind_layer is not None:
                        token_embedding = np.array(
                            [l['values'] for l in feature['layers'] if l['index'] == ind_layer])
                    else:
                        token_embedding = np.concatenate([l['values']for l in feature['layers'] if l['index'] in conc_layers])

                    if token_embeddings is None:
                        token_embeddings = token_embedding
                    else:
                        token_embeddings = np.vstack((token_embeddings, token_embedding))

                    new_question_token.append(feature['token'])
            if len(new_question_token) != token_embeddings.shape[0]:
                print(30 * '*')
                print('********** Size of token question embeddings {} has problem in {} checkpoint **********'.format(question_indx, checkpoint))
                print(30 * '*')
            new_question_tokens.append(new_question_token)
            if question_embeddings is None:
                question_embeddings = token_embeddings
            else:
                question_embeddings = np.vstack((question_embeddings, token_embeddings))

            # # if args['change_shape']:
            # #     question_embedding = np.expand_dims(question_embedding, axis=1)

            # print('Question {} is processed. It has {} tokens and embedding shape is {} so {}'.format(question_indx,
            #                                                                                           org_tokens_len,
            #                                                                                           token_embeddings.shape[0],
            #                                                                                           org_tokens_len == token_embeddings.shape[0]))
        print('Question_embeddings shape: {}'.format(question_embeddings.shape))
        UTIL.save_as_pickle(new_question_tokens, new_question_tokens_path)
        UTIL.dump_embeddings(question_embeddings, contextualized_questions_with_token_file_path)
        print('Questions are dumped')
        del jsons
    paragraph_embeddings = None
    new_paragraph_tokens = []
    if os.path.exists(contextualized_paragraphs_with_token_file_path):
        paragraph_embeddings = UTIL.load_embeddings(contextualized_paragraphs_with_token_file_path)
        new_paragraph_tokens = UTIL.load_from_pickle(new_paragraph_tokens_path)
    else:
        file_names = get_file_names(paragraphs_folder_path, file_name_splitter, bert_extension)
        tokenized_paragraphs_size = test_size[1] if test_size is not None else len(tokenized_paragraphs)
        checkpoint=None
        for paragraph_indx in tqdm(range(tokenized_paragraphs_size)):
            paragraph_bert_index = paragraph_indx + 1
            file_name, remaining_index_to_pass_this_file = find_file_name(paragraph_bert_index, file_names)
            if remaining_index_to_pass_this_file >= 0:
               jsons = UTIL.load_bert_jsons_from_single_file(os.path.join(paragraphs_folder_path, file_name))
               if paragraph_indx >0:
                   checkpoint = paragraph_indx
            if checkpoint is not None:
                paragraph_indx = paragraph_indx - checkpoint

            new_paragraph_token = []
            try:
                for line_index, json in UTIL.reversedEnumerate(jsons[paragraph_indx]):
                    # 0 and -1 token indexes belong to [CLS, SEP] we are ignoring them.
                    json['features'].pop(0)
                    json['features'].pop(-1)

                    # filter out the non-contributional tokens from the list.
                    features = [x for x in json['features'] if not x['token'].startswith("##")]
                    token_embeddings = None
                    for feature_index, feature in UTIL.reversedEnumerate(features):
                        if line_index > 0 and feature_index <= args.window_length:
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

                        new_paragraph_token.append(feature['token'])
                if len(new_paragraph_token) != token_embeddings.shape[0]:
                    print(30 * '*')
                    print('********** Size of token paragraph embeddings {} has problem in {} checkpoint**********'.format(paragraph_indx, checkpoint))
                    print(30 * '*')

                new_paragraph_tokens.append(new_paragraph_token)
                #print('Token Size : {}'.format(sum([len(sentence) for sentence in new_paragraph_tokens])))
            except:
                pass

            if paragraph_embeddings is None:
                paragraph_embeddings = token_embeddings
            else:
                paragraph_embeddings = np.vstack((paragraph_embeddings, token_embeddings))

            # print('Question {} is processed. It has {} tokens and embedding shape is {} so {}'.format(question_indx,
            #                                                                                           org_tokens_len,
            #                                                                                           token_embeddings.shape[
            #                                                                                               0],
            #                                                                                           org_tokens_len ==
            #                                                                                           token_embeddings.shape[
            #                                                                                               0]))
        print('Paragraph_embeddings shape: {}'.format(paragraph_embeddings.shape))
        UTIL.save_as_pickle(new_paragraph_tokens, new_paragraph_tokens_path)
        UTIL.dump_embeddings(paragraph_embeddings, contextualized_paragraphs_with_token_file_path)
        print('Paragraphs are dumped')
        del jsons

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
                for partition in range(0,idf_vec.shape[0], args.partition_size):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args.partition_size, :]
                    temp_idf_vec = idf_vec[partition:partition + args.partition_size, :].reshape(-1,1)
                    #temp_doc_embeddings = temp_doc_embeddings[:,0,:]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    temp_weighted_token_embeddings = np.multiply(temp_idf_vec, temp_doc_embeddings)
                    UTIL.dump_embeddings(temp_weighted_token_embeddings, calculated_token_embeddings_file_path.replace('@@', str(partition_counter)).replace('##', 'idf'))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition, partition + args.partition_size))
        else:
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
                for partition in range(0, len(corpus_as_tokens), args.partition_size):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args.partition_size, :]
                    #temp_doc_embeddings = temp_doc_embeddings[:, 0, :]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    UTIL.dump_embeddings(temp_doc_embeddings, calculated_token_embeddings_file_path.replace('@@', str(partition_counter)).replace('##', ''))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition,
                                                                                          partition + args.partition_size))
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

    print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

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
