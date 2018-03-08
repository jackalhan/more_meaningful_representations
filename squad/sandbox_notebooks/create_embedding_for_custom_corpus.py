'''
Create a database of all the squad paragraphs labelled with the ELMO embeddings.
'''

import os
import h5py
import json
import pandas as pd
from bilm import dump_bilm_embeddings, dump_token_embeddings, TokenBatcher
#import nltk
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf
#from allennlp.data.tokenizers import WordTokenizer
import spacy
nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


#dataset_type = 'train'
dataset_type = 'dev'
dataset_version = 'v1.1'

index_field = ['Unnamed: 0']

# required files
_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)
modeldir = os.path.join(_basepath, 'model')

_squad_file_name = '{}-{}.json'.format(dataset_type, dataset_version)
squad_file = os.path.join(datadir, _squad_file_name)

_vocab_file_name = 'voc.txt'
vocab_file = os.path.join(datadir, _vocab_file_name)

_options_file_name = 'elmo_2x4096_512_2048cnn_2xhighway_weights.json'
_weight_file_name = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

# _options_file_name = 'options.json'
# _weight_file_name = 'lm_weights.hdf5'

options_file = os.path.join(modeldir, _options_file_name)
weight_file = os.path.join(modeldir, _weight_file_name)

_embedding_paragraph_file_as_h5py_name = 'elmo_paragraph_embeddings.hdf5'
embedding_paragraph_file_as_h5py = os.path.join(datadir, _embedding_paragraph_file_as_h5py_name)

_embedding_question_file_as_h5py_name = 'elmo_question_embeddings.hdf5'
embedding_question_file_as_h5py = os.path.join(datadir, _embedding_question_file_as_h5py_name)

# _embedding_file_as_text_name = 'elmo_embeddings.txt'
# embedding_file_as_text = os.path.join(datadir, _embedding_file_as_text_name)

_dataset_file_name = 'data.txt'
dataset_file = os.path.join(datadir, _dataset_file_name)

_titles_file_name = '{}_titles.csv'.format(dataset_type)
titles_file = os.path.join(datadir, _titles_file_name)

_paragraphs_file_name = '{}_paragraphs.csv'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraphs_file_name = '{}_paragraphs.csv'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)
_paragraphs_file_name_as_txt = '{}_paragraphs.txt'.format(dataset_type)
paragraphs_file_as_txt = os.path.join(datadir, _paragraphs_file_name_as_txt)

_questions_file_name = '{}_questions.csv'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)
_questions_file_name_as_txt = '{}_questions.txt'.format(dataset_type)
questions_file_as_txt = os.path.join(datadir, _questions_file_name_as_txt)

_qas_file_name = '{}_qas.csv'.format(dataset_type)
qas_file = os.path.join(datadir, _qas_file_name)

# # Data preparation:
# if not (os.path.exists(titles_file) and
#         os.path.exists(paragraphs_file) and
#         os.path.exists(questions_file) and
#         os.path.exists(qas_file)):
#
#     # Read Dataset From Json File
#     with open(squad_file, 'r') as _squad:
#         squad = json.load(_squad)
#
#     # Parse, titles and contents from the data
#     titles = []
#     paragraphs = []
#     questions = []
#     qas = []
#     _i_para, _i_qas, _i_as = 0, 0, 0
#     for _i_titles, _titles in enumerate(squad['data']):
#         titles.append(_titles['title'])
#         for _paragraph in _titles['paragraphs']:
#             paragraphs.append(_paragraph['context'].replace('\n', ' '))
#             for _qas in _paragraph['qas']:
#                 questions.append((_qas['question'].replace('\n', ' '), _qas['id']))
#                 for _as in _qas['answers']:
#                     qas.append((_i_titles, _i_para, _i_qas, _qas['id'], _as['text'], _as['answer_start']))
#                     _i_as += 1
#                 _i_qas += 1
#             _i_para+=1
#
#
#
#     # convert to dataframe and save them as csv
#     df_titles = pd.DataFrame(data=titles, columns=['Title'])
#     df_titles.to_csv(titles_file)
#
#     df_paragraphs = pd.DataFrame(data=paragraphs, columns=['Paragraph'])
#     df_paragraphs.to_csv(paragraphs_file)
#
#     df_questions = pd.DataFrame(data=questions, columns=['Question', 'Q_id'])
#     df_questions.to_csv(questions_file)
#
#     df_qas = pd.DataFrame(data=qas, columns=['Title_Id', 'Paragraph_Id', 'Question_Id', 'Question_Orj_Id', 'Answer', 'Answer_Start'])
#     df_qas.to_csv(qas_file)
#
# else:
#     df_titles = pd.read_csv(titles_file).set_index(index_field)
#     df_paragraphs = pd.read_csv(paragraphs_file).set_index(index_field)
#     df_questions = pd.read_csv(questions_file).set_index(index_field)
#     df_qas = pd.read_csv(qas_file).set_index(index_field)
#
# all_tokens = set(['<S>', '</S>', '<UNK>'])
#
#
# # ALLEN NLP
# # tokenized_context = [WordTokenizer().tokenize(_)[0] for _ in df_paragraphs['Paragraph']] #0 is added because of getting the tokens instead of their ids
# # tokenized_questions= [WordTokenizer().tokenize(_)[0] for _ in df_questions['Question']]
#
# # SPACY
# tokenized_context = [word_tokenize(_) for _ in df_paragraphs['Paragraph']] #0 is added because of getting the tokens instead of their ids
# tokenized_questions= [word_tokenize(_) for _ in df_questions['Question']]
#
# with open(paragraphs_file_as_txt, 'w') as fout:
#     for sentence in tokenized_context:
#         fout.write(' '.join(sentence) + '\n')
#
# with open(questions_file_as_txt, 'w') as fout:
#     for sentence in tokenized_questions:
#         fout.write(' '.join(sentence) + '\n')
#
# # Create a vocab from tokens
# for sentence in tokenized_context + tokenized_questions:
#     for token in sentence:
#         if token.strip():
#             all_tokens.add(token)
#
# with open(vocab_file, 'w') as voc:
#     voc.write('\n'.join(all_tokens))
#
#
# ## ------------------------------ CREATE EMBEDDINGS USING GIVEN WEIGHTS----------------------------------------------- ##
# ## ------------------------------ BEGIN              ----------------------------------------------- ##
# # Dump the embeddings to a file. Run this once for your dataset.
# Paragraphs


print('Obtaining paragraphs and questions')
_paragraphs_file_name_as_txt = '{}_paragraphs.txt'
paragraphs_file_as_txt = os.path.join(datadir, _paragraphs_file_name_as_txt.format(dataset_type))
sanity_paragraphs_file_as_txt = os.path.join(datadir, _paragraphs_file_name_as_txt.format('sanity_' + dataset_type ))

_questions_file_name_as_txt = '{}_questions.txt'
questions_file_as_txt = os.path.join(datadir, _questions_file_name_as_txt.format(dataset_type))
sanity_questions_file_as_txt = os.path.join(datadir, _questions_file_name_as_txt.format('sanity_' + dataset_type ))
_qas_file_name = '{}_qas.csv'.format('sanity_' + dataset_type )
qas_file = os.path.join(datadir, _qas_file_name)
_voc_file_name_as_txt = '{}_voc.txt'
voc_file_name_as_txt = os.path.join(datadir, _voc_file_name_as_txt.format('sanity_' + dataset_type ))
#
# paragraphs = []
# questions = []
# p_look_up = []
# q_look_up = []
# with open(sanity_paragraphs_file_as_txt, 'r') as fp_in, open(sanity_questions_file_as_txt, 'r') as fq_in:
#     for i, line in enumerate(fp_in):
#         paragraphs.append(line.replace('\n', ' '))
#         p_look_up.append((i, line.replace('\n','')))
#     for i, line in enumerate(fq_in):
#         questions.append(line.replace('\n', ' '))
#         q_look_up.append((i, line.replace('\n','')))
# print('Done')
# df_p_look_up = pd.DataFrame(data=p_look_up, columns=['id', 'paragraph']).set_index('id')
# df_q_look_up = pd.DataFrame(data=q_look_up, columns=['id', 'question']).set_index('id')
# tokenized_paragraphs = [word_tokenize(_) for _ in paragraphs]
# tokenized_questions = [word_tokenize(_) for _ in questions]



dump_bilm_embeddings(
    voc_file_name_as_txt, sanity_paragraphs_file_as_txt, options_file, weight_file, os.path.join(datadir, embedding_paragraph_file_as_h5py)
)
print('Done Paragraphs!')

#questions
dump_bilm_embeddings(
    voc_file_name_as_txt, sanity_questions_file_as_txt, options_file, weight_file, os.path.join(datadir, embedding_question_file_as_h5py)
)
print('Done Questions!')


# # PYTORCH ALLENNLP------------------------------------
#
from allennlp.commands.elmo import ElmoEmbedder
ee = ElmoEmbedder(options_file, weight_file)
print('ee loaded')

with open(sanity_paragraphs_file_as_txt, 'r') as fp_in, open(sanity_questions_file_as_txt, 'r') as fq_in:
    ee.embed_file(fp_in, os.path.join(datadir, embedding_paragraph_file_as_h5py))
    print('Done Paragraphs!')
    ee.embed_file(fq_in, os.path.join(datadir, embedding_question_file_as_h5py))
    print('Done Questions!')

#
# # Load the embeddings from the file -- here the 2nd sentence.
# with h5py.File(os.path.join(datadir, embedding_file), 'r') as fin:
#     second_sentence_embeddings = fin['1'][...]

## ------------------------------ END              ----------------------------------------------- ##