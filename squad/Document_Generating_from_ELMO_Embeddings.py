import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import sys
import numpy as np
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

NEW_API_ELMO={"is_inject_idf":False,
      "root_path": "ELMO",
      "embedding_paragraphs_path": "paragraphs",
      "embedding_paragraphs_file_pattern": "{}_question_embeddings_ELMO_@@".format(dataset_type),
      "contextualized_paragraphs_embeddings_with_token": '{}_contextualized_paragraphs_embeddings_with_token.hdf5'.format(dataset_type),
      "embedding_questions_path": "questions",
      "embedding_questions_file_pattern": "{}_paragraph_embedding_ELMO_@@".format(dataset_type),
      "contextualized_questions_embeddings_with_token": '{}_contextualized_questions_embeddings_with_token.hdf5'.format(dataset_type),
       "is_paragraphs_listed_after_questions":False,
      "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token.hdf5'.format(dataset_type),
      "change_shape": False,
      "weights_arguments": [1],
      'questions_file': '{}_question_document_embeddings.hdf5'.format(dataset_type),
      'paragraphs_file': '{}_paragraph_document_embeddings.hdf5'.format(dataset_type),
      'is_calculate_recalls': True,
      'recall_file_path': '{}_recalls_weights_@@.csv'.format(dataset_type)
      }

OLD_API_ELMO={"is_inject_idf":False,
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
      'questions_file': '{}_question_document_embeddings.hdf5'.format(dataset_type),
      'paragraphs_file': '{}_paragraph_document_embeddings.hdf5'.format(dataset_type),
      'is_calculate_recalls': False,
      'recall_file_path': '{}_recalls_weights_@@.csv'.format(dataset_type)
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
print(100 * '*')
print('Parsing Started')
start = datetime.datetime.now()

word_counter, char_counter = Counter(), Counter()
examples, eval, questions, paragraphs, q_to_ps = UTIL.process_squad_file(squad_file.format(dataset_type),
                                                                        dataset_type,
                                                                        word_counter,
                                                                        char_counter)

print('# of Paragraphs in {} : {}'.format(dataset_type, len(paragraphs)))
print('# of Questions in {} : {}'.format(dataset_type, len(questions)))
print('# of Q_to_P {} : {}'.format(dataset_type, len(q_to_ps)))

print(20 * '-')
print('Paragraphs: Tokenization and Saving Tokenization Started in {}'.format(dataset_type))
tokenized_paragraphs = UTIL.tokenize_contexts(paragraphs)
paragraphs_nontokenized = [" ".join(context) for context in tokenized_paragraphs]
print('# of Tokenized Paragraphs in {} : {}'.format(dataset_type, len(tokenized_paragraphs)))
print(20 * '-')
print('Questions: Tokenization and Saving Tokenization Started in {}'.format(dataset_type))
tokenized_questions = UTIL.tokenize_contexts(questions)
questions_nontokenized = [" ".join(context) for context in tokenized_questions]
print('# of Tokenized Questions in {} : {}'.format(dataset_type, len(tokenized_questions)))
#
# if is_dump_during_execution:
#     UTIL.dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file.format(dataset_type))
#     UTIL.dump_tokenized_contexts(tokenized_questions, questions_file.format(dataset_type))
#     UTIL.dump_mapping_data(q_to_ps, mapping_file.format(dataset_type))
end = datetime.datetime.now()
print('Parsing Ended in {} minutes'.format((end - start).seconds / 60))
print(100 * '*')
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
print(100 * '*')
print('Index of tokens in each document is getting identified....')
start = datetime.datetime.now()
document_embedding_guideline = defaultdict()
corpus_as_tokens = []
for i, sentence in enumerate(tokenized_questions + tokenized_paragraphs):
    document_embedding_guideline[i] = defaultdict()
    document_embedding_guideline[i]['start_index'] = len(corpus_as_tokens)
    document_embedding_guideline[i]['end_index'] = len(corpus_as_tokens) + len(sentence)
    if i >= len(tokenized_questions):
        document_embedding_guideline[i]['type'] = 'p'
    else:
        document_embedding_guideline[i]['type'] = 'q'
    for token in sentence:
        corpus_as_tokens.append(token)

# UTIL.save_as_pickle(document_embedding_guideline, token_embeddings_guideline_file)
# UTIL.save_as_pickle(corpus_as_tokens, tokens_ordered_file)
# del document_embedding_guideline
# del corpus_as_tokens
print("Total tokens in the corpus: {}".format(len(corpus_as_tokens)))
end = datetime.datetime.now()
print('Index of tokens in each document is getting saved in {} minutes'.format((end - start).seconds / 60))
print(100 * '*')
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
        print('Question {} is processed'.format(question_indx))
    print('Question_embeddings shape: {}'.format(question_embeddings.shape))
    UTIL.dump_embeddings(question_embeddings, os.path.join(root_folder_path, args['contextualized_questions_embeddings_with_token']))
    print('Questions are dumped')



paragraphs_folder_path = root_folder_path if args["embedding_paragraphs_path"] is None else os.path.join(root_folder_path, args["embedding_paragraphs_path"])
paragraph_embeddings = None
if os.path.exists(os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token'])):
    paragraph_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token']))
else:

    if args["is_paragraphs_listed_after_questions"]:
        paragraph_range = range(len(tokenized_questions) + len(tokenized_paragraphs), question_indx)
    else:
        paragraph_range = range(len(tokenized_paragraphs))
    for paragraph_indx in paragraph_range:
        p_file_path = os.path.join(paragraphs_folder_path, args['embedding_paragraphs_file_pattern'].replace('@@', str(paragraph_indx)))
        paragraph_embedding= UTIL.load_embeddings(p_file_path)
        if args['change_shape']:
            paragraph_embedding = np.expand_dims(paragraph_embedding, axis=1)
        if paragraph_embeddings is None:
            paragraph_embeddings = paragraph_embedding
        else:
            paragraph_embeddings = np.vstack((paragraph_embeddings,paragraph_embedding))
        print('Paragraph {} is processed'.format(paragraph_indx))
    print('Paragraph_embeddings shape: {}'.format(paragraph_embeddings.shape))
    UTIL.dump_embeddings(paragraph_embeddings, os.path.join(root_folder_path, args['contextualized_paragraphs_embeddings_with_token']))
    print('Paragraphs are dumped')

if os.path.exists(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token'])):
    document_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']))
else:
    document_embeddings = np.vstack((question_embeddings, paragraph_embeddings ))
    UTIL.dump_embeddings(paragraph_embeddings, os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']))
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
    weighted_token_embeddings = np.multiply(idf_vec, document_embeddings)
    end = datetime.datetime.now()
    print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
else:
    print('IDF is skipped')
    _type = 'only'
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

print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

WM = np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
questions_embeddings, paragraphs_embeddings = UTIL.token_to_document_embeddings(tokenized_questions,
                                                                            tokenized_paragraphs,
                                                                            weighted_token_embeddings,
                                                                            document_embedding_guideline,
                                                                            WM)
if args['is_calculate_recalls']:
    print('Recalls are getting calculated')
    calculate_similarity_and_dump(paragraphs_embeddings,
                                  questions_embeddings,
                                  q_to_ps,
                                  len(questions),
                                  os.path.join(root_folder_path,
                                               args['recall_file_path']).replace('@@', '_'.join(args['weights_arguments']))
                                  )
    print('Recalls are calculated')

questions_elmo_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[1]))
UTIL.dump_embeddings(questions_elmo_embeddings, os.path.join(root_folder_path, args['questions_file']))
paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                           (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
UTIL.dump_embeddings(paragraphs_elmo_embeddings, os.path.join(root_folder_path, args['paragraphs_file']))
print('Weighted are applied')
"""
******************************************************************************************************************
******************************************************************************************************************
END : WEIGHTED ARE GETTING APPLIED TO TOKEN EMBEDDINGS
******************************************************************************************************************
******************************************************************************************************************
"""