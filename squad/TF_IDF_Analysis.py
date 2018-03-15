import math
import spacy
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import h5py
nlp = spacy.blank("en")

tokenize = lambda doc: [token.text for token in nlp(doc)]


dataset_type = 'dev'
dataset_version = 'v1.1'

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_paragraphs_file_name = '{}_paragraphs_tfidf.txt'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embeddings_tfidf.hdf5'.format(dataset_type)
paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

_questions_file_name = '{}_questions_tfidf.txt'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)

_question_embeddings_file_name = '{}_question_embeddings_tfidf.hdf5'.format(dataset_type)
question_embeddings_file = os.path.join(datadir, _question_embeddings_file_name)


_neighbors_file_name = '{}_neighbors_tfidf.csv'.format(dataset_type)
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

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    return 1 + math.log(tokenized_document.count(term))

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = None#tokenize_contexts(documents)
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#tfidf_representation = tfidf(all_documents)


#SCIKIT-LEARN IMPLEMENTATION

from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
paragraphs, questions, q_to_p = read_squad_data(squad_file)
inventory = paragraphs + questions
print('Total items :', len(inventory))
print('Paragraphs :', len(inventory[0: len(paragraphs)]))
print('Questions :', len(inventory[len(paragraphs): len(inventory)]))
tf_idfs = sklearn_tfidf.fit_transform(inventory)
paragraphs_embeddings = np.array(tf_idfs[0: len(paragraphs)].toarray().tolist())
questions_embeddings = np.array(tf_idfs[len(paragraphs): len(inventory)].toarray().tolist())
print(tf_idfs.toarray()[0].tolist())
print(inventory[0])

with h5py.File(question_embeddings_file, 'w') as fout:
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        ds = fout.create_dataset('{}'.format(_id),
                                 _q_embedding.shape, dtype='float32',
                                 data=_q_embedding
                                 )


with h5py.File(paragraph_embeddings_file, 'w') as fout:
    for _id, _p_embedding in enumerate(tqdm(paragraphs_embeddings, total=len(paragraphs_embeddings))):
        ds = fout.create_dataset('{}'.format(_id),
                                 _p_embedding.shape, dtype='float32',
                                 data=_p_embedding
                                 )



slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]

_s = slices[0] # option 1

neighbor_list = []
print('Nearest Neighbors: Starting')
for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
    _q_embedding = np.array([_q_embedding])
    sk_sim = cosine_similarity(_q_embedding,paragraphs_embeddings)[0]
    neighbors = np.argsort(-sk_sim)
    for _, neighbor_id in enumerate(neighbors):
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
                                                         'actual_paragrraph_cos_similarity'
                                                         ])
df_neighbors.to_csv(neighbors_file, index=False)
print('Nearest Neighbors: Completed')


