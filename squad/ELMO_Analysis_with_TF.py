import os
from bilm.elmo import ElmoEmbedder
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
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

_questions_file_name = '{}_questions.txt'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)

_neighbors_file_name = '{}_neighbors.csv'.format(dataset_type)
neighbors_file = os.path.join(datadir, _neighbors_file_name)

_squad_file_name = '{}-{}.json'.format(dataset_type, dataset_version)
squad_file = os.path.join(datadir, _squad_file_name)

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


print('Squad Data: Reading')
paragraphs, questions, q_to_p = read_squad_data(squad_file)
print('# of Paragraphs : {}'.format(len(paragraphs)))
print('# of Questions : {}'.format(len(questions)))
print('# of Q_to_P : {}'.format(len(q_to_p)))
print('Squad Data: Reading Completed')

print(20* '-')
print('Paragraphs: Tokenized')
tokenized_paragraphs = tokenize_contexts(paragraphs)
dump_tokenized_contexts(tokenized_paragraphs, questions_file)
print('# of Tokenized Paragraphs: {}'.format(len(tokenized_paragraphs)))
print('Paragraphs: Tokenizing is Completed')

print(20* '-')
print('Questions: Tokenized')
tokenized_questions = tokenize_contexts(questions)
dump_tokenized_contexts(tokenized_questions, paragraphs_file)
print('# of Tokenized Questions: {}'.format(len(tokenized_questions)))
print('Questions: Tokenizing is Completed')


ee = ElmoEmbedder()

slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]

_s = slices[0] # option 1

neighbor_list = []
print('Processing : {}'.format(_s))

print(20* '-')
print('Paragraphs: Embedding')
paragraphs_embeddings = np.asarray(ee.list_to_embeddings(tokenized_paragraphs, _s['slice_index']))
paragraphs_embeddings = np.reshape(paragraphs_embeddings, (paragraphs_embeddings.shape[0], ee.dims))
print('# of Embedded Paragraphs: {}'.format(paragraphs_embeddings.shape[0]))
print('Paragraphs: Embedding is completed')

print(20* '-')
print('Question: Embedding')
questions_embeddings = np.asarray(ee.list_to_embeddings(tokenized_questions, _s['slice_index']))
questions_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], ee.dims))
print('# of Embedded Questions: {}'.format(questions_embeddings.shape[0]))
print('Question: Embedding is completed')

print('Nearest Neighbors: Starting')
for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
    _q_embedding = np.array([_q_embedding])
    sk_sim = cosine_similarity(_q_embedding,paragraphs_embeddings)[0]
    neighbors = np.argsort(-sk_sim)
    for _, neighbor_id in enumerate(neighbors[0:5]):
            neighbor_list.append((_s['slice_type'],
                                  _id,
                                  neighbor_id,
                                  _+1,
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
                                                         'neighbor_cos_similarity'
                                                         ])
df_neighbors.to_csv(neighbors_file, index=False)
print('Nearest Neighbors: Completed')


