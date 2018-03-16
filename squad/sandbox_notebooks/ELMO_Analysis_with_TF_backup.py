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

slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]

s = slices[0] # option 1

neighbor_list = []
#print('Processing : {}'.format(_s))

print(20* '-')


# # # TOKEN EMBEDDINGS START
# print('Paragraphs + Questions: Token ELMO Embeddings Generator')
# start = datetime.datetime.now()
# word_embeddings, voc, vocab_file = ee.list_to_token_embeddings(  batch=tokenized_paragraphs+tokenized_questions+tokenized_test_questions+tokenized_test_paragraphs,
#                                                                  inject_tfidf=False,
#                                                                  tfidf_smooth_idf=False,
#                                                                  tfidf_sublinear_tf=False,
#                                                                  inject_mean_glove=False,
#                                                                  required_tfidf_mean=False,
#                                                                  glove_path=glove_file,
#                                                                  outfile_to_dump=word_embeddings_file
#                                                                 )
# del word_embeddings
# del voc
# end = datetime.datetime.now()
# print('Paragraphs + Questions: Token ELMO Embeddings Generator is completed in {} minutes'.format((end-start).seconds/60))
#
#
# print('Paragraphs + Questions: ELMO Computation Started')
# bigstart = datetime.datetime.now()
# tf.reset_default_graph()
# ## Now we can do inference.
# # Create a TokenBatcher to map text to token ids.
# batcher = TokenBatcher(vocab_file)
#
# # Input placeholders to the biLM.
# context_token_ids = tf.placeholder('int32', shape=(None, None))
# question_token_ids = tf.placeholder('int32', shape=(None, None))
#
# # Build the biLM graph.
# bilm = BidirectionalLanguageModel(
#     ee.options_file_path,
#     ee.weight_file_path,
#     use_character_inputs=False,
#     embedding_weight_file=word_embeddings_file
# )
#
# # Get ops to compute the LM embeddings.
# context_embeddings_op = bilm(context_token_ids)
# question_embeddings_op = bilm(question_token_ids)
#
# # Get an op to compute ELMo (weighted average of the internal biLM layers)
# # Our SQuAD model includes ELMo at both the input and output layers
# # of the task GRU, so we need 4x ELMo representations for the question
# # and context at each of the input and output.
# # We use the same ELMo weights for both the question and context
# # at each of the input and output.
# elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
# with tf.variable_scope('', reuse=True):
#     # the reuse=True scope reuses weights from the context for the question
#     elmo_question_input = weight_layers(
#         'input', question_embeddings_op, l2_coef=0.0
#     )
#
# elmo_context_output = weight_layers(
#     'output', context_embeddings_op, l2_coef=0.0
# )
# with tf.variable_scope('', reuse=True):
#     # the reuse=True scope reuses weights from the context for the question
#     elmo_question_output = weight_layers(
#         'output', question_embeddings_op, l2_coef=0.0
#     )
#
#
# with tf.Session() as sess:
#     # It is necessary to initialize variables once before running inference.
#     sess.run(tf.global_variables_initializer())
#
#     # Create batches of data.
#     context_ids = batcher.batch_sentences(tokenized_paragraphs)
#     question_ids = batcher.batch_sentences(tokenized_questions)
#
#
#     print('Computing ELMO representation for Inputs is started')
#     start = datetime.datetime.now()
#     # Compute ELMo representations (here for the input only, for simplicity).
#     elmo_context_input_, elmo_question_input_ = sess.run(
#         [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
#         feed_dict={context_token_ids: context_ids,
#                    question_token_ids: question_ids}
#     )
#     end = datetime.datetime.now()
#     print('Computing ELMO representation for Inputs is ended in {} minutes'.format((end-start).seconds/60))
#
#
#
#     print('Processing : {}'.format(s))
#     print('Writing Input Embeddings to Dataset is started')
#     start = datetime.datetime.now()
#     with h5py.File(os.path.join(datadir, 'input_paragraph_embeddings.hdf5'), 'w') as fp_out, h5py.File(os.path.join(datadir, 'input_question_embeddings.hdf5'), 'w') as fq_out:
#         print('Paragraph Input Embeddings for {} is getting written'.format(s))
#         elmo_paragraph_input_embeddings = []
#         for i, _contents in enumerate(tqdm(elmo_context_input_, total=len(elmo_context_input_))):
#             _contents_mean = np.apply_over_axes(np.mean, _contents, (0))
#             elmo_paragraph_input_embeddings.append(_contents_mean)
#             ds = fp_out.create_dataset(
#                 '{}'.format(i),
#                 _contents_mean.shape, dtype='float32',
#                 data=_contents_mean
#             )
#         elmo_paragraph_input_embeddings = np.asarray(elmo_paragraph_input_embeddings)
#         elmo_paragraph_input_embeddings = np.reshape(elmo_paragraph_input_embeddings, (elmo_paragraph_input_embeddings.shape[0], ee.dims))
#
#         elmo_question_input_embeddings = []
#         print('Question Input Embeddings for {} is getting written'.format(s))
#         for i, _contents in enumerate(tqdm(elmo_question_input_, total=len(elmo_question_input_))):
#             _contents_mean = np.apply_over_axes(np.mean, _contents, (0))
#             elmo_question_input_embeddings.append(_contents_mean)
#             ds = fq_out.create_dataset(
#                 '{}'.format(i),
#                 _contents_mean.shape, dtype='float32',
#                 data=_contents_mean
#             )
#         elmo_question_input_embeddings = np.asarray(elmo_question_input_embeddings)
#         elmo_question_input_embeddings = np.reshape(elmo_question_input_embeddings,
#                                                      (elmo_question_input_embeddings.shape[0], ee.dims))
#     end = datetime.datetime.now()
#     print('Writing Input Embeddings to Dataset is ended in {} minutes'.format((end - start).seconds/60))
#     del elmo_context_input_
#     del elmo_question_input_
#     print('Calculating Similarities for Input Embeddings is started')
#     start = datetime.datetime.now()
#     calculate_similarity(elmo_paragraph_input_embeddings, elmo_question_input_embeddings,s['slice_type'],ee.dims, q_to_p, os.path.join(datadir, 'input_neighbors.csv'))
#     end = datetime.datetime.now()
#     del elmo_paragraph_input_embeddings
#     del elmo_question_input_embeddings
#     print('Calculating Similarities for Input Embeddings is ended in {} minutes.'.format((end-start).seconds/60))
#         #------------------------------------------------
#         # ------------------------------------------------
#
#
#
#     print('Computing ELMO representation for Outputs is started')
#     start = datetime.datetime.now()
#     elmo_context_output_, elmo_question_output_ = sess.run(
#         [elmo_context_output['weighted_op'], elmo_question_output['weighted_op']],
#         feed_dict={context_token_ids: context_ids,
#                    question_token_ids: question_ids}
#     )
#     end = datetime.datetime.now()
#     print('Computing ELMO representation for Outputs is ended in {} minutes'.format((end - start).seconds/60))
#
#     print('Writing Output Embeddings to Dataset is started')
#     start = datetime.datetime.now()
#     with h5py.File(os.path.join(datadir, 'output_paragraph_embeddings.hdf5'), 'w') as fp_out, h5py.File(
#             os.path.join(datadir, 'output_question_embeddings.hdf5'),'w') as fq_out:
#         print('Paragraph Output Embeddings for {} is getting written')
#         elmo_paragraph_output_embeddings = []
#         for i, _contents in enumerate(tqdm(elmo_context_output_, total=len(elmo_context_output_))):
#             _contents_mean = np.apply_over_axes(np.mean, _contents, (0))
#             elmo_paragraph_output_embeddings.append(_contents_mean)
#             ds = fp_out.create_dataset(
#                 '{}'.format(i),
#                 _contents_mean.shape, dtype='float32',
#                 data=_contents_mean
#             )
#         elmo_paragraph_output_embeddings = np.asarray(elmo_paragraph_output_embeddings)
#         elmo_paragraph_output_embeddings = np.reshape(elmo_paragraph_output_embeddings,
#                                                     (elmo_paragraph_output_embeddings.shape[0], ee.dims))
#
#
#         print('Question Output Embeddings for {} is getting written')
#         elmo_question_output_embeddings = []
#         for i, _contents in enumerate(tqdm(elmo_question_output_, total=len(elmo_question_output_))):
#             _contents_mean = np.apply_over_axes(np.mean, _contents, (0))
#             elmo_question_output_embeddings.append(_contents_mean)
#             ds = fq_out.create_dataset(
#                 '{}'.format(i),
#                 _contents_mean.shape, dtype='float32',
#                 data=_contents_mean
#             )
#         elmo_question_output_embeddings = np.asarray(elmo_question_output_embeddings)
#         elmo_question_output_embeddings = np.reshape(elmo_question_output_embeddings,
#                                                       (elmo_question_output_embeddings.shape[0], ee.dims))
#     end = datetime.datetime.now()
#     print('Writing Output Embeddings to Dataset is ended in {} minutes'.format((end - start).seconds/60))
#     del elmo_context_output_
#     del elmo_question_output_
#
#     print('Calculating Similarities for Output Embeddings is started')
#     start = datetime.datetime.now()
#     calculate_similarity(elmo_paragraph_output_embeddings, elmo_question_output_embeddings, s['slice_type'],
#                          ee.dims, q_to_p, os.path.join(datadir, 'output_neighbors.csv'))
#     end = datetime.datetime.now()
#     del elmo_paragraph_output_embeddings
#     del elmo_question_output_embeddings
#     print('Calculating Similarities for Output Embeddings is ended in {} minutes.'.format((end - start).seconds/60))
#
# bigend = datetime.datetime.now()
# print('Paragraphs + Questions: ELMO Computation Ended in {} minutes'.format((bigend-bigstart).seconds/60))

# TOKEN EMBEDDINGS END



    # CHAR EMBEDDINGS END

s = slices[0]

# token_tfidf_weights = None
# tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=True, sublinear_tf=True)
# tfidf.fit(tokenized_test_questions+tokenized_test_paragraphs+tokenized_questions+tokenized_paragraphs)
# tfidf_paragraphs = np.array(tfidf.transform(tokenized_paragraphs).toarray().tolist())
# tfidf_questions = np.array(tfidf.transform(tokenized_questions).toarray().tolist())
# CHAR EMBEDDINGS START
print('Paragraphs: Embedding')
paragraphs_embeddings = np.asarray(ee.list_to_embeddings_with_dump(tokenized_paragraphs, s['slice_index'], paragraph_embeddings_file))
paragraphs_embeddings = np.reshape(paragraphs_embeddings, (paragraphs_embeddings.shape[0], ee.dims))
print('# of Embedded Paragraphs: {}'.format(paragraphs_embeddings.shape[0]))
print('Paragraphs: Embedding is completed')

print(20* '-')
print('Question: Embedding')
questions_embeddings = np.asarray(ee.list_to_embeddings_with_dump(tokenized_questions, s['slice_index'], question_embeddings_file))
questions_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], ee.dims))
print('# of Embedded Questions: {}'.format(questions_embeddings.shape[0]))
print('Question: Embedding is completed')

print('Nearest Neighbors: Starting')
start = datetime.datetime.now()
calculate_similarity(paragraphs_embeddings, questions_embeddings, s['slice_type'],
                     ee.dims, q_to_p, os.path.join(datadir, 'output_neighbors.csv'))
end = datetime.datetime.now()
print('Nearest Neighbors: Completed')
# CHAR EMBEDDINGS END

