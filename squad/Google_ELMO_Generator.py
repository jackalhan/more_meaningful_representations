import datetime
from collections import Counter, defaultdict
import tensorflow as tf
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = TRAIN
is_dump_during_execution = True
is_inject_idf = True
is_filtered_by_answers_from_rnet = False

# ELMO EMBEDDINGS #
is_elmo_embeddings= True

# USE EMBEDDINGS #
is_use_embedding = False

################ CONFIGURATIONS #################


_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

paragraphs_dir = UTIL.create_dir(os.path.join(datadir, 'paragraphs'))
questions_dir = UTIL.create_dir(os.path.join(datadir, 'questions'))

_paragraphs_file_name = '{}_paragraphs.txt'
paragraphs_file = os.path.join(paragraphs_dir, _paragraphs_file_name)

_questions_file_name = '{}_questions.txt'
questions_file = os.path.join(questions_dir, _questions_file_name)

_mapping_file_name = '{}_q_to_p_mappings.csv'
mapping_file = os.path.join(questions_dir, _mapping_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embedding_@.hdf5'.format(dataset_type)
paragraph_embedding_file = os.path.join(paragraphs_dir, _paragraph_embeddings_file_name)

_question_embeddings_file_name = '{}_question_embeddings_@.hdf5'.format(dataset_type)
question_embeddings_file = os.path.join(questions_dir, _question_embeddings_file_name)

_token_embeddings_guideline_file_name = '{}_token_embeddings_guideline.pkl'.format(dataset_type)
token_embeddings_guideline_file = os.path.join(datadir, _token_embeddings_guideline_file_name)

_tokens_ordered_file_name = '{}_tokens_ordered.pkl'.format(dataset_type)
tokens_ordered_file = os.path.join(datadir, _tokens_ordered_file_name)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

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

if is_dump_during_execution:
    UTIL.dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file.format(dataset_type))
    UTIL.dump_tokenized_contexts(tokenized_questions, questions_file.format(dataset_type))
    UTIL.dump_mapping_data(q_to_ps, mapping_file.format(dataset_type))
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

UTIL.save_as_pickle(document_embedding_guideline, token_embeddings_guideline_file)
UTIL.save_as_pickle(corpus_as_tokens, tokens_ordered_file)
del document_embedding_guideline
del corpus_as_tokens
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
START: GOOGLE ELMO EMBEDDINGS
******************************************************************************************************************
******************************************************************************************************************
"""
if is_elmo_embeddings:

    print(100 * '*')
    print('Generating ELMO Embeddings from Google just started....')
    start = datetime.datetime.now()
    for document_type in ['question','paragraph']:
        index = 0
        if document_type == 'question':
            tokenized = tokenized_questions
            reset_every_iter = 25
            embedding_file = question_embeddings_file
        else:
            tokenized = tokenized_paragraphs
            reset_every_iter = 20
            embedding_file = paragraph_embedding_file

        while True:
            tf.reset_default_graph()
            elmo_embed = UTIL.load_module("https://tfhub.dev/google/elmo/2", trainable=True)
            tf.logging.set_verbosity(tf.logging.ERROR)
            begin_index = index * reset_every_iter
            end_index = begin_index + reset_every_iter
            print('Processing {} from {} to {}'.format(document_type, begin_index, end_index))
            if begin_index <= len(tokenized):
                with tf.Session() as session:
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                    for i, each_document in enumerate(tqdm(tokenized[begin_index:end_index],
                                                           total=len(tokenized[begin_index:end_index])), begin_index):

                        d = session.run(elmo_embed(
                            inputs={
                                "tokens": [each_document],
                                "sequence_len": [len(each_document)]
                            },
                            signature="tokens",
                            as_dict=True)['lstm_outputs1'])
                        d = d[0,:,:]
                        UTIL.dump_embeddings(d, embedding_file.replace('@', 'LSTM1_' + str(i)))

                        d = session.run(elmo_embed(
                            inputs={
                                "tokens": [each_document],
                                "sequence_len": [len(each_document)]
                            },
                            signature="tokens",
                            as_dict=True)['lstm_outputs2'])
                        d = d[0, :, :]
                        UTIL.dump_embeddings(d, embedding_file.replace('@', 'LSTM2_' + str(i)))

                        d = session.run(elmo_embed(
                            inputs={
                                "tokens": [each_document],
                                "sequence_len": [len(each_document)]
                            },
                            signature="tokens",
                            as_dict=True)['elmo'])
                        d = d[0, :, :]
                        UTIL.dump_embeddings(d, embedding_file.replace('@', 'ELMO_' + str(i)))

                index += 1
            else:
                break
    end = datetime.datetime.now()
    print('ELMO Embeddings from Google are generated in {} minutes'.format((end - start).seconds / 60))
    print(100 * '*')
"""
******************************************************************************************************************
******************************************************************************************************************
END : GOOGLE ELMO EMBEDDINGS
******************************************************************************************************************
******************************************************************************************************************
"""