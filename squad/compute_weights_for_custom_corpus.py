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
from allennlp.data.tokenizers import WordTokenizer

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

_embedding_file_as_h5py_name = 'elmo_embeddings.hdf5'
embedding_file_as_h5py = os.path.join(datadir, _embedding_file_as_h5py_name)
_embedding_file_as_text_name = 'elmo_embeddings.txt'
embedding_file_as_text = os.path.join(datadir, _embedding_file_as_text_name)

_dataset_file_name = 'data.txt'
dataset_file = os.path.join(datadir, _dataset_file_name)

_titles_file_name = '{}_titles.csv'.format(dataset_type)
titles_file = os.path.join(datadir, _titles_file_name)

_paragraphs_file_name = '{}_paragraphs.csv'.format(dataset_type)
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_questions_file_name = '{}_questions.csv'.format(dataset_type)
questions_file = os.path.join(datadir, _questions_file_name)

_qas_file_name = '{}_qas.csv'.format(dataset_type)
qas_file = os.path.join(datadir, _qas_file_name)

# Data preparation:
if not (os.path.exists(titles_file) and
        os.path.exists(paragraphs_file) and
        os.path.exists(questions_file) and
        os.path.exists(qas_file)):

    # Read Dataset From Json File
    with open(squad_file, 'r') as _squad:
        squad = json.load(_squad)

    # Parse, titles and contents from the data
    titles = []
    paragraphs = []
    questions = []
    qas = []
    for _i_titles, _titles in enumerate(squad['data']):
        titles.append(_titles['title'])
        for _i_para, _paragraph in enumerate(_titles['paragraphs']):
            paragraphs.append(_paragraph['context'].replace('\n', ' '))
            for _i_qas, _qas in enumerate(_paragraph['qas']):
                questions.append((_qas['question'].replace('\n', ' '), _qas['id']))
                for _i_as, _as in enumerate(_qas['answers']):
                    qas.append((_i_titles, _i_para, _i_qas, _qas['id'], _as['text'], _as['answer_start']))

    # convert to dataframe and save them as csv
    df_titles = pd.DataFrame(data=titles, columns=['Title'])
    df_titles.to_csv(titles_file)

    df_paragraphs = pd.DataFrame(data=paragraphs, columns=['Paragraph'])
    df_paragraphs.to_csv(paragraphs_file)

    df_questions = pd.DataFrame(data=questions, columns=['Question', 'Q_id'])
    df_questions.to_csv(questions_file)

    df_qas = pd.DataFrame(data=qas, columns=['Title_Id', 'Paragraph_Id', 'Question_Id', 'Question_Orj_Id', 'Answer', 'Answer_Start'])
    df_qas.to_csv(qas_file)

else:
    df_titles = pd.read_csv(titles_file).set_index(index_field)
    df_paragraphs = pd.read_csv(paragraphs_file).set_index(index_field)
    df_questions = pd.read_csv(questions_file).set_index(index_field)
    df_qas = pd.read_csv(qas_file).set_index(index_field)

all_tokens = set(['<S>', '</S>', '<UNK>'])

# Tokenize paragraphs with NLTK
# tokenized_context = [nltk.word_tokenize(_) for _ in paragraphs]
# Tokenize paragraphs with allenNLP
tokenized_context = [WordTokenizer().tokenize(_)[0] for _ in df_paragraphs['Paragraph']] #0 is added because of getting the tokens instead of their ids
tokenized_questions = [WordTokenizer().tokenize(_)[0] for _ in df_questions['Question']]

# if not os.path.exists(vocab_file):

# Create a vocab from tokens
for sentence in tokenized_context + tokenized_questions:
    for token in sentence:
        if token.strip():
            all_tokens.add(token)

with open(vocab_file, 'w') as voc:
    voc.write('\n'.join(all_tokens))

# # Create the dataset file.
# with open(dataset_file, 'w') as fout:
#     for paragraph in tokenized_context:
#         fout.write(' '.join(paragraph) + '\n')
# else:
#     with open(vocab_file, 'r') as voc:
#         line = voc.readline()
#         all_tokens.add(line)


## ------------------------------ COMPUTE WEIGHTS FOR OUR DATASET ----------------------------------------------- ##
## ------------------------------ BEGIN              ----------------------------------------------- ##
dump_token_embeddings(
    vocab_file, options_file, weight_file, embedding_file_as_h5py
)
tf.reset_default_graph()

# Create a TokenBatcher to map text to token ids
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=embedding_file_as_h5py
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)
question_embeddings_op = bilm(question_token_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_questions)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_token_ids: context_ids,
                   question_token_ids: question_ids}
    )

## ------------------------------ END              ----------------------------------------------- ##