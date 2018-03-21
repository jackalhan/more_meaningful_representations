import datetime
import os
from collections import defaultdict
import pickle
from bilm.elmo import ElmoEmbedder
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
from shutil import copyfile
nlp = spacy.blank("en")
encoding="utf-8"
tokenize = lambda doc: [token.text for token in nlp(doc)]
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

dataset_type = 'train'
dataset_version = 'v1.1'

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_paragraphs_file_name = '{}_paragraphs.txt'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embeddings.hdf5'.format(dataset_type)
paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

_token_embeddings_file_name = '{}_token_embeddings.hdf5'.format(dataset_type)
token_embeddings_file= os.path.join(datadir, _token_embeddings_file_name )

_token_embeddings_guideline_file_name = '{}_token_embeddings_guideline.pkl'.format(dataset_type)
token_embeddings_guideline_file = os.path.join(datadir, _token_embeddings_guideline_file_name)

_questions_file_name = '{}_questions.txt'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)

_question_embeddings_file_name = '{}_question_embeddings.hdf5'.format(dataset_type)
question_embeddings_file = os.path.join(datadir, _question_embeddings_file_name)

_word_embeddings_file_name = '{}_word_embeddings.hdf5'.format(dataset_type)
word_embeddings_file = os.path.join(datadir, _word_embeddings_file_name)

_neighbors_file_name = '{}_neighbors.csv'.format(dataset_type)
neighbors_file = os.path.join(datadir, _neighbors_file_name)

_voc_file_name = '{}_voc.txt'.format(dataset_type)
voc_file_name = os.path.join(datadir, _voc_file_name)

_squad_file_name = '{}-{}.json'.format(dataset_type, dataset_version)
squad_file = os.path.join(datadir, _squad_file_name)

_squad_other_file_name = '{}-{}.json'.format('dev', dataset_version)
squad_other_file = os.path.join(datadir, _squad_other_file_name)

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

def calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, slice_type, q_to_p, outfile):
    neighbor_list = []
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
    return df_neighbors

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def get_elmo_embeddings(tokenized_questions, tokenized_paragraphs, token_embeddings_guideline_file,
                        token_embeddings_file, voc_file_name):
    document_embedding_guideline = defaultdict()
    if not os.path.exists(token_embeddings_guideline_file):
        #########################
        ## use word embedding ##
        # ee = ElmoEmbedder(embedding_file=word_embeddings_file)
        ##########################
        ## use char encoding embedding ##
        ee = ElmoEmbedder()
        ##########################
        voc_file = ee.batch_to_vocs(tokenized_questions + tokenized_paragraphs)
        copyfile(voc_file, voc_file_name)
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

        with open(token_embeddings_guideline_file, 'wb') as handle:
            pickle.dump(document_embedding_guideline, handle, protocol=pickle.HIGHEST_PROTOCOL)

        document_embeddings = ee.list_to_lazy_embeddings_with_dump(corpus_as_tokens, token_embeddings_file)
    else:
        with open(token_embeddings_guideline_file, 'rb') as handle:
            document_embedding_guideline = pickle.load(handle)

        with h5py.File(token_embeddings_file, 'r') as fin:
            document_embeddings = fin['embeddings'][...]


    return document_embeddings, document_embedding_guideline

def read_file(file_name):
    with open(file_name) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def dump_embeddings(embeddings, outfile_to_dump):
    with h5py.File(outfile_to_dump, 'w') as fout:
        ds = fout.create_dataset(
            'embeddings',
            embeddings.shape, dtype='float32',
            data=embeddings
        )

def create_idf_matrix(tokenized_questions, tokenized_paragraphs, token2idfweight):
    idf_matrix = []
    for sentence in tokenized_questions + tokenized_paragraphs:
        for word in sentence:
            idf_matrix.append(token2idfweight[word])

    idf_matrix = np.asarray(idf_matrix)
    idf_matrix = idf_matrix.reshape(idf_matrix.shape[0], 1,1)
    return idf_matrix

def token_to_document_embeddings(tokenized_questions, tokenized_paragraphs,token_embeddings, token_embeddings_guideline):
    questions_embeddings = []
    paragraphs_embeddings = []
    for _ in tqdm(range(len(tokenized_questions + tokenized_paragraphs))):
        str_index = token_embeddings_guideline[_]['start_index']
        end_index = token_embeddings_guideline[_]['end_index']
        d_type = token_embeddings_guideline[_]['type']

        if d_type == 'q':
            questions_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_question_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))
        else:
            paragraphs_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_paragraph_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))
    del token_embeddings

    questions_embeddings = np.asarray(questions_embeddings)
    paragraphs_embeddings = np.asarray(paragraphs_embeddings)

    return questions_embeddings, paragraphs_embeddings


print('Squad Data: Reading Dev Started')
start = datetime.datetime.now()
paragraphs, questions, q_to_p = read_squad_data(squad_other_file)
paragraphs_test, questions_test, q_to_p_test = read_squad_data(squad_file)
end = datetime.datetime.now()
print('# of Paragraphs : {}'.format(len(paragraphs_test)))
print('# of Questions : {}'.format(len(questions_test)))
print('# of Q_to_P : {}'.format(len(q_to_p_test)))
print('Squad Data: Reading Dev Ended in {} minutes'.format((end-start).seconds/60))


print(20* '-')
print('Paragraphs: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_paragraphs = tokenize_contexts(paragraphs)
tokenized_test_paragraphs = tokenize_contexts(paragraphs_test)
dump_tokenized_contexts(tokenized_test_paragraphs, paragraphs_file)
end = datetime.datetime.now()
print('# of Tokenized Paragraphs: {}'.format(len(tokenized_test_paragraphs)))
print('Paragraphs: Tokenization and Saving Tokenization  is Completed in {} minutes'.format((end-start).seconds/60))

print(20* '-')
print('Questions: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_questions = tokenize_contexts(questions)
tokenized_test_questions = tokenize_contexts(questions_test)
dump_tokenized_contexts(tokenized_test_questions,questions_file)
end = datetime.datetime.now()
print('# of Tokenized Questions: {}'.format(len(tokenized_test_questions)))
print('Questions: Tokenization and Saving Tokenization  is Completed in {} minutes'.format((end-start).seconds/60))



slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]

s = slices[0] # option 1

print('Processing : {}'.format(s))
print(20* '-')

questions_nontokenized = [" ".join(context) for context in tokenized_questions]
paragraphs_nontokenized = [" ".join(context) for context in tokenized_paragraphs]
questions_test_nontokenized = [" ".join(context) for context in tokenized_test_questions]
paragraphs_test_nontokenized = [" ".join(context) for context in tokenized_test_paragraphs]


# tfidf_paragraphs = np.array(tfidf.transform(tokenized_paragraphs).toarray().tolist())
# tfidf_questions = np.array(tfidf.transform(tokenized_questions).toarray().tolist())

print(20* '-')
print('ELMO Token Embeddings is started')
start = datetime.datetime.now()
token_embeddings, token_embeddings_guideline = get_elmo_embeddings(tokenized_test_questions, tokenized_test_paragraphs, token_embeddings_guideline_file, token_embeddings_file, voc_file_name)
end = datetime.datetime.now()
print('ELMO Token Embeddings is ended in {} minutes'.format((end-start).seconds/60))
#WEIGHT MATRIX FOR TUNING
# a = .3
# b = 1-a
# WM = np.array([1, a, b]).reshape((1,3,1))

#IDF MATRIX SHAPE OF [x, 1, k, 1], where x = number of documents, k = max length of document
#IDFM =

tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenize)
tfidf.fit(questions_nontokenized+paragraphs_nontokenized+questions_test_nontokenized+paragraphs_test_nontokenized)
max_idf = max(tfidf.idf_)
token2idfweight = defaultdict(
    lambda: max_idf,
    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
idf_vec = create_idf_matrix(tokenized_test_questions, tokenized_test_paragraphs, token2idfweight)
#IDF_WEIGHTED
idf_weighted_token_embeddings = np.multiply(idf_vec, token_embeddings)

print(20* '-')
_a_b_c_s = []
# _a_b_c_s.append([0,0,1])
# _a_b_c_s.append([0, 1, 0])
_a_b_c_s.append([1, 0, 0])
# while len(_a_b_c_s) < 10:
#     x = np.random.dirichlet(np.ones(3), size=1).tolist()
#     x_ = [float("{:1.2f}".format(_x)) for _x in x[0]]
#     total_x_ = sum(x_)
#     if total_x_ == 1:
#         _a_b_c_s.append(x_)
#         _a_b_c_s = sort_and_deduplicate(_a_b_c_s)

for _token_embed_pack in [(idf_weighted_token_embeddings, 'with_idf'), (token_embeddings, 'only')]:
    _token_embed = _token_embed_pack[0]
    _type = _token_embed_pack[1]

    start = datetime.datetime.now()
    print('ELMO Embeddings is started for "{}" type'.format(_type))

    for _a_b_c in _a_b_c_s:
        print('Weight {}'.format(_a_b_c))
        questions_embeddings, paragraphs_embeddings = token_to_document_embeddings(tokenized_test_questions,
                                                                                   tokenized_test_paragraphs, _token_embed,
                                                                                   token_embeddings_guideline)
        # YES TUNE
        WM = np.array([_a_b_c[0], _a_b_c[1], _a_b_c[2]]).reshape((1, 3, 1))
        questions_embeddings = np.multiply(questions_embeddings, WM)
        paragraphs_embeddings = np.multiply(paragraphs_embeddings, WM)

        questions_embeddings = np.mean(questions_embeddings, axis=1)
        paragraphs_embeddings = np.mean(paragraphs_embeddings, axis=1)
        print('Nearest Neighbors: Starting')
        calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, s['slice_type'], q_to_p_test,
                                      os.path.join(datadir, 'elmo_{}_weights_a_{}_b_{}_c_{}_output_neighbors.csv'.format(_type, _a_b_c[0], _a_b_c[1], _a_b_c[2])))
        print('Nearest Neighbors: Completed')
    end = datetime.datetime.now()
    print('ELMO Embeddings is completed in {} minutes for "{}" type'.format((end - start).seconds / 60, _type))
    print(20 * '-')
# questions_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[2]))
# paragraphs_embeddings = np.reshape(paragraphs_embeddings, (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[2]))



# print('Nearest Neighbors: Starting')
# start = datetime.datetime.now()
# calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, s['slice_type'], q_to_p, os.path.join(datadir, 'elmo_with_idf_output_neighbors.csv'))
# #calculate_similarity_and_dump(idf_injected_paragraph_embeddings, idf_injected_question_embeddings, s['slice_type'], q_to_p, os.path.join(datadir, 'elmo_with_idf_output_neighbors.csv'))
# end = datetime.datetime.now()
# print('Nearest Neighbors: Completed')


