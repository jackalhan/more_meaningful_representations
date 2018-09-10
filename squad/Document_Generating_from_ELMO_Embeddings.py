import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import sys
import numpy as np
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = DEV

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

NEW_API_ELMO={"is_inject_idf":True,
"load_data_partially": False,
"partition_size": None,
"calculated_idf_token_embeddings_file": '{}_contextualized_document_embeddings_with_token_and_idf_@@.hdf5'.format(dataset_type),
      "root_path": "ELMO_CONTEXT_NEW_API_EMBEDDINGS",
      "embedding_paragraphs_path": "paragraphs",
      "embedding_paragraphs_file_pattern": "{}_paragraph_embedding_LSTM1_@@.hdf5".format(dataset_type),
      "contextualized_paragraphs_embeddings_with_token": '{}_contextualized_paragraphs_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
      "embedding_questions_path": "questions",
      "embedding_questions_file_pattern": "{}_question_embeddings_LSTM1_@@.hdf5".format(dataset_type),
      "contextualized_questions_embeddings_with_token": '{}_contextualized_questions_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
       "is_paragraphs_listed_after_questions":False,
      "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token_LSTM1.hdf5'.format(dataset_type),
      "change_shape": False,
      "weights_arguments": [1],
      'questions_file': '{}_question_document_embeddings_LSTM1_@@.hdf5'.format(dataset_type),
      'paragraphs_file': '{}_paragraph_document_embeddings_LSTM1_@@.hdf5'.format(dataset_type),
      'is_calculate_recalls': True,
      'recall_file_path': '{}_recalls_weights_LSTM1_@@_###.csv'.format(dataset_type)
      }

OLD_API_ELMO={"is_inject_idf":False,
              "load_data_partially": True,
              "partition_size": 100000,
              "calculated_idf_token_embeddings_file": '{}_contextualized_document_embeddings_with_token_@@.hdf5'.format(dataset_type),
      "root_path": "ELMO_CONTEXT_OLD_API_EMBEDDINGS",
      "embedding_paragraphs_path": None,
      "embedding_paragraphs_file_pattern": "{}_token_embeddings_old_api_doc_@@.hdf5".format(dataset_type),
      "contextualized_paragraphs_embeddings_with_token": '{}_contextualized_paragraphs_embeddings_with_token.hdf5'.format(dataset_type),
      "embedding_questions_path": None,
      "embedding_questions_file_pattern": "{}_token_embeddings_old_api_doc_@@.hdf5".format(dataset_type),
      "contextualized_questions_embeddings_with_token": '{}_contextualized_questions_embeddings_with_token.hdf5'.format(dataset_type),
      "is_paragraphs_listed_after_questions":True,
      "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token.hdf5'.format(dataset_type),
      "change_shape": False,
      "weights_arguments": [1, 0, 0],
      'questions_file': '{}_question_document_embeddings_@@.hdf5'.format(dataset_type),
      'paragraphs_file': '{}_paragraph_document_embeddings_@@.hdf5'.format(dataset_type),
      'is_calculate_recalls': False,
      'recall_file_path': '{}_recalls_weights_@@_###.csv'.format(dataset_type)
      }

args = OLD_API_ELMO


################ CONFIGURATIONS #################


################ ALGOS #################
def calculate_recall_at_n(ns, data, number_of_questions):
    recalls = []
    for i in ns:
        total_number = len(data[(data['nearest_neighbor_order'] <= i) & (data['ground_truth'] == True) ])
        recalls.append((i, total_number, total_number/number_of_questions))
    return recalls

def calculate_similarity_and_dump(paragraphs_embeddings,
                                  questions_embeddings,
                                  q_to_p,
                                  number_of_questions,
                                  outfile):
    neighbor_list = []
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        for _, neighbor_id in enumerate(neighbors):
            neighbor_list.append((_id,
                                  neighbor_id,
                                  (q_to_p[_id] == neighbor_id),
                                  True,
                                  sk_sim[neighbor_id],
                                  _,
                                  ))

    columns = ['question', 'paragraph', 'ground_truth', 'is_model_answered_correctly',
               'cosine_score', 'nearest_neighbor_order']
    df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list, columns=columns)
    df_neighbor_within_paragraph = df_neighbor_within_paragraph[
        df_neighbor_within_paragraph['is_model_answered_correctly'] == True]

    df_neighbor_within_paragraph.to_csv(outfile.replace('###', ''), index=False)
    recall_ns = [1, 2, 5, 10, 20, 50]
    recall_columns = ['n', 'number_of_true', 'normalized_recalls']
    df_neighbor_within_paragraph_recalls = pd.DataFrame(data=calculate_recall_at_n(recall_ns,
                                                                                   df_neighbor_within_paragraph,
                                                                                   number_of_questions)
                                                        , columns=recall_columns
                                                        )

    df_neighbor_within_paragraph_recalls.to_csv(outfile.replace('###', 'recalls'),
                                                index=False)
################ ALGOS #################


"""
******************************************************************************************************************
******************************************************************************************************************
START: PARSING FILE
******************************************************************************************************************
******************************************************************************************************************
"""
tokenized_questions, tokenized_paragraphs, questions_nontokenized, paragraphs_nontokenized = UTIL.prepare_squad_objects(squad_file.format(dataset_type),dataset_type)
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
root_folder_path = os.path.join(datadir, args["root_path"])
document_embeddings = None
questions_folder_path = root_folder_path if args["embedding_questions_path"] is None else os.path.join(root_folder_path, args["embedding_questions_path"])
question_embeddings = None
if os.path.exists(os.path.join(root_folder_path, args['contextualized_questions_embeddings_with_token'])):
    question_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_questions_embeddings_with_token']))
else:
    for question_indx in range(len(tokenized_questions)):
        q_file_path = os.path.join(questions_folder_path, args['embedding_questions_file_pattern'].replace('@@', str(question_indx)))
        question_embedding= UTIL.load_embeddings(q_file_path)
        if args['change_shape']:
            question_embedding = np.expand_dims(question_embedding, axis=1)
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



paragraphs_folder_path = root_folder_path if args["embedding_paragraphs_path"] is None else os.path.join(root_folder_path, args["embedding_paragraphs_path"])
paragraph_embeddings = None
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
                temp_idf_vec = idf_vec[partition:partition + args["partition_size"], :,:]
                temp_weighted_token_embeddings = np.multiply(temp_idf_vec, temp_doc_embeddings)
                UTIL.dump_embeddings(temp_weighted_token_embeddings, os.path.join(root_folder_path, args['calculated_idf_token_embeddings_file'].replace('@@', str(partition_counter))))
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
                UTIL.dump_embeddings(temp_doc_embeddings, os.path.join(root_folder_path, args[
                    'calculated_idf_token_embeddings_file'].replace('@@', str(partition_counter))))
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
            'calculated_idf_token_embeddings_file'].replace('@@', str(partition))))
        if weighted_token_embeddings is None:
            weighted_token_embeddings = temp_weighted_token_embeddings
        else:
            weighted_token_embeddings = np.vstack((weighted_token_embeddings, temp_weighted_token_embeddings))
        print("Partition {} is loaded".format(partition))

print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

WM = np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
questions_embeddings, paragraphs_embeddings = UTIL.token_to_document_embeddings(tokenized_questions,
                                                                            tokenized_paragraphs,
                                                                            weighted_token_embeddings,
                                                                            document_embedding_guideline,
                                                                            WM)
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