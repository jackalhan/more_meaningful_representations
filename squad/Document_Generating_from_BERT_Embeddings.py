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

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', default=3, type=int, help="-1: no windows, else should be selected from the given range 1-512. it should be at least 1 lower than truncate_length")
    #parser.add_argument('--truncate_length', default=512, type=int, help="-1: unlimited, else length should be selected from the given range 1-512.")
    parser.add_argument('--data_path', help="path to the source files, this folder(dev/train) must contain questions/paragraphs folder for embeddings")
    #parser.add_argument('--dataset_type', default='dev', help="the reason of the execution is train or dev")
    parser.add_argument('--squad_formatted_file', help='squad formatted dataset')
    parser.add_argument('--is_inject_idf', default=False, help="whether inject idf or not to the weights")
    parser.add_argument('--is_convert_to_tokens', default=True, help="convert document embeddings to tokenized embeddings")
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
    document_embeddings = None
    question_embeddings = None
    paragraph_embeddings = None
    questions_folder_path = os.path.join(args.data_path, 'questions')
    paragraphs_folder_path = os.path.join(args.data_path, 'paragraphs')
    contextualized_questions_with_token_path = UTIL.create_dir(
        os.path.join(args.data_path, "contextualized_questions_embeddings_with_token.hdf5"))
    contextualized_paragraphs_with_token_path = UTIL.create_dir(
        os.path.join(args.data_path, "contextualized_paragraphs_embeddings_with_token.hdf5"))
    ################ CONFIGURATIONS #################


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
    START: DOCUMENT-TOKEN GUIDELINE
    ******************************************************************************************************************
    ******************************************************************************************************************
    """

    document_embedding_guideline, corpus_as_tokens = UTIL.generate_document_embedding_guideline(tokenized_questions, tokenized_paragraphs)

    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    END: DOCUMENT-TOKEN GUIDELINE
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
    bert_extension = ".json"
    file_name_splitter = '_'
    if not args.is_convert_to_tokens:
        question_embeddings = UTIL.load_embeddings(contextualized_questions_with_token_path)
    else:
        file_names = get_file_names(questions_folder_path, file_name_splitter, bert_extension)
        for question_indx in range(len(tokenized_questions)):
            question_bert_index = question_indx + 1
            file_name, remaining_index_to_pass_this_file = find_file_name(question_bert_index, file_names)
            if remaining_index_to_pass_this_file >= 0:
                raw_embeddings = UTIL.load_multijsons_from_single_file(os.path.join(questions_folder_path, file_name))
            cleaned_sub_embeddings = None
            for raw_embedding in raw_embeddings[question_indx]:
                _raw_embedding =raw_embedding['features']
                if cleaned_sub_embeddings is None:
                    question_embeddings = question_embedding
                else:
                    question_embeddings = np.vstack((question_embeddings, question_embedding))
            # if args['change_shape']:
            #     question_embedding = np.expand_dims(question_embedding, axis=1)
            if question_embeddings is None:
                question_embeddings = question_embedding
            else:
                question_embeddings = np.vstack((question_embeddings,question_embedding))
            if len(tokenized_questions[question_indx]) != question_embedding.shape[0]:
                print(30 * '*')
                print('********** Question {} has problem **********'.format(question_indx))
                print(30 * '*')

            print('Question {} is processed. It has {} tokens and embedding shape is {} so {}'.format(question_indx,
                                                                                                      len(tokenized_questions[question_indx]),
                                                                                                      question_embedding.shape[0],
                                                                                                      len(tokenized_questions[
                                                                                                          question_indx]) == question_embedding.shape[0]))
        print('Question_embeddings shape: {}'.format(question_embeddings.shape))
        UTIL.dump_embeddings(question_embeddings, os.path.join(root_folder_path, args['contextualized_questions_embeddings_with_token']))
        print('Questions are dumped')




    if os.path.exists(os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token'])):
        print('contextualized_paragraphs_embeddings_with_token found')
        paragraph_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token']))
    else:

        if args["is_paragraphs_listed_after_questions"]:
            paragraph_range = range(len(tokenized_questions), len(tokenized_questions) + len(tokenized_paragraphs))
        else:
            paragraph_range = range(len(tokenized_paragraphs))

        print('paragraph_range {}'.format(paragraph_range))
        for par_tokenized_indx, paragraph_indx in enumerate(paragraph_range):
            p_file_path = os.path.join(paragraphs_folder_path, args['embedding_paragraphs_file_pattern'].replace('@@', str(paragraph_indx)))
            paragraph_embedding= UTIL.load_embeddings(p_file_path)
            if args['change_shape']:
                paragraph_embedding = np.expand_dims(paragraph_embedding, axis=1)
            if paragraph_embeddings is None:
                paragraph_embeddings = paragraph_embedding
            else:
                paragraph_embeddings = np.vstack((paragraph_embeddings,paragraph_embedding))

            if len(tokenized_paragraphs[par_tokenized_indx]) != paragraph_embedding.shape[0]:
                print(30 * '*')
                print('********** Paragraph {} has problem **********'.format(paragraph_indx))
                print(30 * '*')
            print('Paragraph {} is processed. It has {} tokens and embedding shape is {} so {}'.format(paragraph_indx,
                                                                                                      len(tokenized_paragraphs[
                                                                                                          par_tokenized_indx]),
                                                                                                      paragraph_embedding.shape[
                                                                                                          0],
                                                                                                       len(tokenized_paragraphs[
                                                                                                           par_tokenized_indx]) ==
                                                                                                       paragraph_embedding.shape[
                                                                                                      0]))

        print('Paragraph_embeddings shape: {}'.format(paragraph_embeddings.shape))
        UTIL.dump_embeddings(paragraph_embeddings, os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token']))
        print('Paragraphs are dumped')

    if os.path.exists(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token'])):
        if args['load_data_partially'] is not True:
            document_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']))
    else:
        document_embeddings = np.vstack((question_embeddings, paragraph_embeddings ))
        UTIL.dump_embeddings(document_embeddings, os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']))
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

    """
    ******************************************************************************************************************
    ******************************************************************************************************************
    START: IDF
    ******************************************************************************************************************
    ******************************************************************************************************************
    """
    if args['is_inject_idf']:
        print('IDF is going to be calculated')
        nlp = spacy.blank("en")
        tokenize = lambda doc: [token.text for token in nlp(doc)]
        start = datetime.datetime.now()
        token2idfweight, idf_vec = UTIL.transform_to_idf_weigths(tokenized_questions,
                                                                 tokenized_paragraphs,
                                                                 tokenize,
                                                                 questions_nontokenized,
                                                                 paragraphs_nontokenized)
        if args['load_data_partially'] is True:
            with h5py.File(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']), 'r') as fin:
                partition_counter=0
                for partition in range(0,idf_vec.shape[0], args["partition_size"]):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args["partition_size"], :,:]
                    temp_idf_vec = idf_vec[partition:partition + args["partition_size"], :,:].reshape(-1,1)
                    temp_doc_embeddings = temp_doc_embeddings[:,0,:]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    temp_weighted_token_embeddings = np.multiply(temp_idf_vec, temp_doc_embeddings)
                    UTIL.dump_embeddings(temp_weighted_token_embeddings, os.path.join(root_folder_path, args['calculated_token_embeddings_file'].replace('@@', str(partition_counter)).replace('##', 'idf')))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition, partition + args["partition_size"]))
        else:
            weighted_token_embeddings = np.multiply(idf_vec, document_embeddings)
        del idf_vec
        del token2idfweight
        end = datetime.datetime.now()
        print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
    else:
        print('IDF is skipped')
        _type = 'only'
        if args['load_data_partially'] is True:
            with h5py.File(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']),
                           'r') as fin:
                partition_counter = 0
                for partition in range(0, len(corpus_as_tokens), args["partition_size"]):
                    partition_counter += 1
                    temp_doc_embeddings = fin['embeddings'][partition:partition + args["partition_size"], :, :]
                    temp_doc_embeddings = temp_doc_embeddings[:, 0, :]
                    #temp_doc_embeddings = preprocessing.normalize(temp_doc_embeddings, norm='l2')
                    UTIL.dump_embeddings(temp_doc_embeddings, os.path.join(root_folder_path, args[
                        'calculated_token_embeddings_file'].replace('@@', str(partition_counter)).replace('##', '')))
                    print("Partition {} is completed and processed {} - {} tokens".format(partition_counter, partition,
                                                                                          partition + args[
                                                                                              "partition_size"]))
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
    if args['load_data_partially'] is True:
        weighted_token_embeddings = None
        for partition in range(1, partition_counter+1):
            temp_weighted_token_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args[
                'calculated_token_embeddings_file'].replace('@@', str(partition)).replace('##', 'idf' if args['is_inject_idf'] else '')))
            if weighted_token_embeddings is None:
                weighted_token_embeddings = temp_weighted_token_embeddings
            else:
                weighted_token_embeddings = np.vstack((weighted_token_embeddings, temp_weighted_token_embeddings))
            print("Partition {} is loaded".format(partition))

    print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

    WM = None #np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
    questions_embeddings, paragraphs_embeddings = UTIL.token_to_document_embeddings(tokenized_questions,
                                                                                tokenized_paragraphs,
                                                                                weighted_token_embeddings,
                                                                                document_embedding_guideline,
                                                                                WM
                                                                                )
    if args['is_calculate_recalls']:
        print('Recalls are getting calculated')
        if args['is_inject_idf']:
            calculate_similarity_and_dump(paragraphs_embeddings,
                                          questions_embeddings,
                                          q_to_ps,
                                          len(questions),
                                          os.path.join(root_folder_path,
                                                       args['recall_file_path']).replace('@@', 'with_idf' + '_'.join([str(x) for x in args['weights_arguments']]))
                                          )
        else:
            calculate_similarity_and_dump(paragraphs_embeddings,
                                          questions_embeddings,
                                          q_to_ps,
                                          len(questions),
                                          os.path.join(root_folder_path,
                                                       args['recall_file_path']).replace('@@', '_'.join(
                                              [str(x) for x in args['weights_arguments']]))
                                          )
        print('Recalls are calculated')

    if args['is_inject_idf']:
        questions_elmo_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[1]))
        UTIL.dump_embeddings(questions_elmo_embeddings, os.path.join(root_folder_path, args['questions_file'].replace('@@', 'with_idf_'+ '_'.join([str(x) for x in args['weights_arguments']]))))
        paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                                   (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
        UTIL.dump_embeddings(paragraphs_elmo_embeddings, os.path.join(root_folder_path, args['paragraphs_file'].replace('@@', 'with_idf_'+'_'.join([str(x) for x in args['weights_arguments']]))))
    else:
        questions_elmo_embeddings = np.reshape(questions_embeddings,
                                               (questions_embeddings.shape[0], questions_embeddings.shape[1]))
        UTIL.dump_embeddings(questions_elmo_embeddings, os.path.join(root_folder_path, args['questions_file'].replace('@@', '_'.join(
                                                                                                                          [
                                                                                                                              str(
                                                                                                                                  x)
                                                                                                                              for
                                                                                                                              x
                                                                                                                              in
                                                                                                                              args[
                                                                                                                                  'weights_arguments']]))))
        paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                                (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
        UTIL.dump_embeddings(paragraphs_elmo_embeddings, os.path.join(root_folder_path,
                                                                      args['paragraphs_file'].replace('@@', '_'.join(
                                                                          [str(x) for x in args['weights_arguments']]))))

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
