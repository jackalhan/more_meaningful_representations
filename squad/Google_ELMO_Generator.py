import datetime
from collections import Counter, defaultdict
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = DEV

laptop={"batch_question": 20,
        "batch_paragraph": 1,
        }

titanX={"batch_question": 5000,
        "batch_paragraph": 40,
        }

resource=titanX

is_dump_during_execution = False
is_inject_idf = True
is_filtered_by_answers_from_rnet = False

################ CONFIGURATIONS #################


_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

paragraphs_dir = UTIL.create_dir(os.path.join(datadir, 'ELMO', 'paragraphs'))
questions_dir = UTIL.create_dir(os.path.join(datadir, 'ELMO','questions'))

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
if is_dump_during_execution:
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
print(100 * '*')
print('Generating ELMO Embeddings from Google just started....')
start = datetime.datetime.now()
for document_type in ['question','paragraph']:
    counter = 0
    if document_type == 'question':
        begin_index = 0
        documents = questions_nontokenized
        tokenized_documents = tokenized_questions
        reset_every_iter = 3
        batch = resource['batch_question']
        embedding_file = question_embeddings_file
    else:
        begin_index = 0
        documents = paragraphs_nontokenized
        tokenized_documents = tokenized_paragraphs
        reset_every_iter = 3
        batch = resource['batch_paragraph']
        embedding_file = paragraph_embedding_file

    while begin_index <= len(documents)-1:
        if counter % reset_every_iter == 0:
            print('Graph is resetted')
            tf.reset_default_graph()
            elmo_embed = UTIL.load_module("https://tfhub.dev/google/elmo/2", trainable=True)
            tf.logging.set_verbosity(tf.logging.ERROR)

        begin_index = begin_index
        end_index = begin_index + batch
        if end_index > len(documents):
            end_index = len(documents)
        print('Processing {} from {} to {}'.format(document_type, begin_index, end_index))
        with tf.Session() as session:
            print('Session is opened')
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            d1 = session.run(elmo_embed( documents[begin_index:end_index],
                signature="default",
                as_dict=True)['lstm_outputs1'])

            d2 = session.run(elmo_embed( documents[begin_index:end_index],
                signature="default",
                as_dict=True)['lstm_outputs2'])

            delmo= session.run(elmo_embed( documents[begin_index:end_index],
                signature="default",
                as_dict=True)['elmo'])
            # for i, each_document in enumerate(tqdm(tokenized[begin_index:end_index],
            #                                        total=len(tokenized[begin_index:end_index])), begin_index):

            for doc_index, embed_document in enumerate(enumerate(documents[begin_index:end_index]), begin_index):

                try:
                    embed_index, each_document = embed_document
                    _begining = 0
                    _ending = len(tokenized_documents[doc_index])
                    _d1 = d1[embed_index,_begining:_ending,:]
                    _d1 = np.expand_dims(_d1, axis=1)
                    UTIL.dump_embeddings(_d1, embedding_file.replace('@', 'LSTM1_' + str(doc_index)))
                    _d2 = d2[embed_index, _begining:_ending,:]
                    _d2 = np.expand_dims(_d2, axis=1)
                    UTIL.dump_embeddings(_d2, embedding_file.replace('@', 'LSTM2_' + str(doc_index)))
                    _delmo = delmo[embed_index, _begining:_ending, :]
                    _delmo = np.expand_dims(_delmo, axis=1)
                    UTIL.dump_embeddings(_delmo, embedding_file.replace('@', 'ELMO_' + str(doc_index)))
                except Exception as ex:
                    print(ex)
                    print('End of documents')

        counter += 1
        begin_index += batch
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