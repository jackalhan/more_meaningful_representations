import datetime
import os
from bilm.elmo import ElmoEmbedder
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from bilm.data import TokenBatcher
from bilm import BidirectionalLanguageModel, weight_layers
import tensorflow as tf
import spacy
import json
import h5py
nlp = spacy.blank("en")
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

dataset_type = 'dev'
dataset_version = 'v1.1'

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_paragraphs_file_name = '{}_paragraphs.txt'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embeddings.hdf5'.format(dataset_type)
paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

_document_embeddings_file_name = '{}_document_embeddings.hdf5'.format(dataset_type)
document_embeddings_file = os.path.join(datadir, _document_embeddings_file_name)

_questions_file_name = '{}_questions.txt'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)

_question_embeddings_file_name = '{}_question_embeddings.hdf5'.format(dataset_type)
question_embeddings_file = os.path.join(datadir, _question_embeddings_file_name)

_word_embeddings_file_name = '{}_word_embeddings.hdf5'.format(dataset_type)
word_embeddings_file = os.path.join(datadir, _word_embeddings_file_name)

_neighbors_file_name = '{}_neighbors.csv'.format(dataset_type)
neighbors_file = os.path.join(datadir, _neighbors_file_name)

_squad_file_name = '{}-{}.json'.format(dataset_type, dataset_version)
squad_file = os.path.join(datadir, _squad_file_name)

_squad_test_file_name = '{}-{}.json'.format('test', dataset_version)
squad_test_file = os.path.join(datadir, _squad_file_name)

_glove_file_name = 'glove.840B.300d.txt'
glove_file = os.path.join(datadir, _glove_file_name)

def read_squad_data(squad_file_path):

    #Read Dataset From Json File
    with open(squad_file_path, 'r') as _squad:
        squad = json.load(_squad)
    # Parse, titles and contents from the data
    paragraphs = []
    questions = []
    question_to_paragraph = []
    _i_para, _i_qas = 0, 0
    for _i_titles, _titles in enumerate(squad['data']):
        for _paragraph in _titles['paragraphs']:
            paragraphs.append(_paragraph['context'])
            for _qas in _paragraph['qas']:
                questions.append(_qas['question'])
                question_to_paragraph.append(_i_para)
                _i_qas += 1
            _i_para+=1

    return paragraphs, questions, question_to_paragraph

def dump_tokenized_contexts(tokenized_contexts:list, file_path:str):
    with open(file_path, 'w') as fout:
        for context in tokenized_contexts:
            fout.write(' '.join(context) + '\n')


def tokenize_contexts(contexts:list):
    tokenized_context = [word_tokenize(context.strip()) for context in contexts]
    return tokenized_context

def calculate_similarity(paragraphs_embeddings, questions_embeddings, slice_type, dims, q_to_p, outfile):
    # CHAR EMBEDDINGS START
    print('Paragraphs: Embedding')
    paragraphs_embeddings = np.reshape(paragraphs_embeddings, (paragraphs_embeddings.shape[0], dims))
    print('# of Embedded Paragraphs: {}'.format(paragraphs_embeddings.shape[0]))
    print('Paragraphs: Embedding is completed')

    print(20 * '-')
    print('Question: Embedding')
    questions_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], dims))
    print('# of Embedded Questions: {}'.format(questions_embeddings.shape[0]))
    print('Question: Embedding is completed')

    print('Nearest Neighbors: Starting')
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        for _, neighbor_id in enumerate(neighbors):
            neighbor_list.append((slice_type,
                                  _id,
                                  neighbor_id,
                                  _ + 1,
                                  sk_sim[neighbor_id],
                                  q_to_p[_id],
                                  np.where(neighbors == q_to_p[_id])[0][0] + 1,
                                  sk_sim[q_to_p[_id]]
                                  ))
    df_neighbors = pd.DataFrame(data=neighbor_list, columns=['slice_type',
                                                             'question',
                                                             'neighbor_paragraph',
                                                             'neighbor_order',
                                                             'neighbor_cos_similarity',
                                                             'actual_paragraph',
                                                             'actual_paragraph_order',
                                                             'actual_paragrraph_cos_similarity'
                                                             ])
    df_neighbors.to_csv(outfile, index=False)
    print('Nearest Neighbors: Completed')

print('Squad Data: Reading Dev Started')
start = datetime.datetime.now()
paragraphs, questions, q_to_p = read_squad_data(squad_file)
end = datetime.datetime.now()
print('Squad Data: Reading Dev Ended in {} minutes'.format((end-start).seconds/60))

print('Squad Data: Reading Test')
paragraphs_test, questions_test, q_to_p_test = read_squad_data(squad_test_file)

print('# of Paragraphs : {}'.format(len(paragraphs)))
print('# of Questions : {}'.format(len(questions)))
print('# of Q_to_P : {}'.format(len(q_to_p)))
print('Squad Data: Reading Completed')

print(20* '-')
print('Paragraphs: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_paragraphs = tokenize_contexts(paragraphs)
tokenized_test_paragraphs = tokenize_contexts(paragraphs_test)
dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file)
end = datetime.datetime.now()
print('# of Tokenized Paragraphs: {}'.format(len(tokenized_paragraphs)))
print('Paragraphs: Tokenization and Saving Tokenization  is Completed in {} mminutes'.format((end-start).seconds/60))

print(20* '-')
print('Questions: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_questions = tokenize_contexts(questions)
tokenized_test_questions = tokenize_contexts(questions_test)
dump_tokenized_contexts(tokenized_questions,questions_file)
end = datetime.datetime.now()
print('# of Tokenized Questions: {}'.format(len(tokenized_questions)))
print('Questions: Tokenization and Saving Tokenization  is Completed in {} mminutes'.format((end-start).seconds/60))


#########################
## usae word embedding ##
# ee = ElmoEmbedder(embedding_file=word_embeddings_file)
##########################


#########################
## usae char encoding embedding ##
ee = ElmoEmbedder()
##########################
voc_file = ee.batch_to_vocs(tokenized_questions + tokenized_paragraphs)

slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]

s = slices[0] # option 1

neighbor_list = []
#print('Processing : {}'.format(_s))

print(20* '-')


s = slices[0]
# CHAR EMBEDDINGS END

#WEIGHT MATRIX FOR TUNING
WM = np.array([1, 0, 0]).reshape((1,3,1,1))

#IDF MATRIX SHAPE OF [x, 1, k, 1], where x = number of documents, k = max length of document
#IDFM =
# token_tfidf_weights = None
# tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=True, sublinear_tf=True)
# tfidf.fit(tokenized_test_questions+tokenized_test_paragraphs+tokenized_questions+tokenized_paragraphs)
# tfidf_paragraphs = np.array(tfidf.transform(tokenized_paragraphs).toarray().tolist())
# tfidf_questions = np.array(tfidf.transform(tokenized_questions).toarray().tolist())
# CHAR EMBEDDINGS START
print('Embeddings is started')
document_embeddings = ee.list_to_embeddings_with_dump(tokenized_questions, document_embeddings_file)
#document_embeddings = np.reshape(document_embeddings, (document_embeddings.shape[0], ee.dims))
print('# of Embedded Paragraphs: {}'.format(document_embeddings.shape[0]))
print('Embedding is completed')


print('Nearest Neighbors: Starting')
start = datetime.datetime.now()
# calculate_similarity(paragraphs_embeddings, questions_embeddings, s['slice_type'],
#                      ee.dims, q_to_p, os.path.join(datadir, 'output_neighbors.csv'))
end = datetime.datetime.now()
print('Nearest Neighbors: Completed')
# CHAR EMBEDDINGS END

