import datetime
from collections import Counter, defaultdict
import tensorflow as tf
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
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

args={"is_inject_idf":False,
      "root_path": "ELMO",
      "embedding_paragraphs_path": "paragraphs",
      "embedding_paragraphs_file_pattern": "{}_question_embeddings_ELMO_@@".format(dataset_type),
      "contexualized_paragraphs_embeddings_with_token": 'contexualized_paragraphs_embeddings_with_token.hdf5',
      "embedding_questions_path": "questions",
      "embedding_questions_file_pattern": "{}_paragraph_embedding_ELMO_@@".format(dataset_type),
      "contexualized_questions_embeddings_with_token": 'contexualized_questions_embeddings_with_token.hdf5',
      "change_shape": True,
      }

#resource=laptop

################ CONFIGURATIONS #################

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


"""
******************************************************************************************************************
******************************************************************************************************************
END: LOAD EMBEDINGS
******************************************************************************************************************
******************************************************************************************************************
"""

root_folder_path = os.path.join(datadir, args["root_path"])
questions_folder_path = os.path.join(root_folder_path, args["embedding_questions_path"])
question_embeddings = None
for question_indx in range(len(tokenized_questions)):
    q_file_path = os.path.join(questions_folder_path, args['embedding_questions_file_pattern'].format(question_indx))
    question_embedding= UTIL.load_embeddings(q_file_path)
    if args['change_shape']:
        question_embedding = np.expand_dims(question_embedding, axis=1)
    if question_embeddings is None:
        question_embeddings = question_embedding
    else:
        question_embeddings = np.vstack((question_embeddings,question_embedding))
    print('Question {} is processed'.format(question_indx))
UTIL.dump_embeddings(question_embeddings, os.path.join(root_folder_path, args['contexualized_questions_embeddings_with_token']))
print('Questions are dumped')

paragraphs_folder_path = os.path.join(root_folder_path, args["embedding_paragraphs_path"])
paragraph_embeddings = None
for paragraph_indx in range(len(tokenized_paragraphs)):
    p_file_path = os.path.join(paragraphs_folder_path, args['embedding_paragraphs_file_pattern'].format(question_indx))
    paragraph_embedding= UTIL.load_embeddings(p_file_path)
    if args['change_shape']:
        paragraph_embeddings = np.expand_dims(paragraph_embedding, axis=1)
    if question_embeddings is None:
        paragraph_embeddings = paragraph_embedding
    else:
        paragraph_embeddings = np.vstack((paragraph_embeddings,paragraph_embedding))
    print('Paragraph {} is processed'.format(paragraph_indx))
UTIL.dump_embeddings(paragraph_embeddings, os.path.join(root_folder_path, args['contexualized_paragraphs_embeddings_with_token']))
print('Paragraphs are dumped')

"""
******************************************************************************************************************
******************************************************************************************************************
START: IDF
******************************************************************************************************************
******************************************************************************************************************
"""
# if args['is_inject_idf']:
#     print('IDF is going to be calculated')
#     nlp = spacy.blank("en")
#     tokenize = lambda doc: [token.text for token in nlp(doc)]
#     start = datetime.datetime.now()
#     token2idfweight, idf_vec = UTIL.transform_to_idf_weigths(tokenized_questions,
#                                                              tokenized_paragraphs,
#                                                              tokenize,
#                                                              questions_nontokenized,
#                                                              paragraphs_nontokenized)
#     weighted_token_embeddings = np.multiply(idf_vec, token_embeddings)
#     end = datetime.datetime.now()
#     print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
# else:
#     print('IDF is skipped')
#     _type = 'only'
#     weighted_token_embeddings = token_embeddings
"""
******************************************************************************************************************
******************************************************************************************************************
END: LOAD IDF
******************************************************************************************************************
******************************************************************************************************************
"""