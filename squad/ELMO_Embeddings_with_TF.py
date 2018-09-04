import datetime
import os
import re
import string
from collections import defaultdict, Counter
from itertools import chain
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import random
import h5py
import math
from math import gcd
import tensorflow as tf
import tensorflow_hub as hub
#from glove import Glove, Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from shutil import copyfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from bilm.elmo import ElmoEmbedder
import helper.utils as UTIL
nlp = spacy.blank("en")
nlp_s = spacy.load('en')
encoding="utf-8"
tokenize = lambda doc: [token.text for token in nlp(doc)]
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
def sentence_segmenter(context):
    _context = nlp_s(context)
    return list(_context.sents)

def pad_list(l, max_num, padding_key):
    return l[:max_num] + [padding_key] * (max_num - len(l))

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total,_i_para  = 0, 0
    questions = []
    paragraphs = []
    question_to_paragraph = []
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            title = article["title"]
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                paragraphs.append(context)
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    questions.append(ques)
                    question_to_paragraph.append(_i_para)
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, 'ques': ques,"answers": answer_texts, "uuid": qa["id"], 'title': title}
                _i_para += 1
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples, questions, paragraphs, question_to_paragraph


# def create_glove_corpus_model(corpus, model_path, window = 10, case_sensitive=True):
#
#     corpus_model = Corpus()
#     if case_sensitive:
#         my_corpus = [[x.lower() for x in _ ]for _ in corpus]
#     corpus_model.fit(my_corpus, window=window)
#     corpus_model.save(model_path)
#     return corpus_model
#
# def create_glove_vector_model(corpus_model, model_path, dims = 100, learning_rate=0.05, epoch = 10,  threads = 10):
#
#     glove_model = Glove(no_components=dims, learning_rate=learning_rate)
#     glove_model.fit(corpus_model.matrix, epochs=epoch,
#               no_threads=threads, verbose=True)
#     glove_model.add_dictionary(corpus_model.dictionary)
#     glove_model.save(model_path)
#     return glove_model

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
                ground_truths = list(map(lambda x: x['text'], _qas['answers']))
                questions.append(_qas['question'])
                question_to_paragraph.append(_i_para)
                _i_qas += 1
            _i_para+=1

    return paragraphs, questions, question_to_paragraph


def any_match_for_paragraph_and_answer(paragraph, answers):
    _paragraph = normalize_answer(paragraph)
    for answer in answers:
        answer = normalize_answer(answer)
        if _paragraph.find(answer) > -1:
            return True
    return False

def any_match_for_answer_and_prediction(prediction, answers):
    _prediction = normalize_answer(prediction)
    for answer in answers:
        answer = normalize_answer(answer)
        if answer == _prediction:
            return True
    return False


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def dump_tokenized_contexts(tokenized_contexts:list, file_path:str):
    with open(file_path, 'w') as fout:
        for context in tokenized_contexts:
            fout.write(' '.join(context) + '\n')

def tokenize_contexts(contexts:list):
    tokenized_context = [word_tokenize(context.strip()) for context in contexts]
    return tokenized_context

def calculate_similarity_and_dump(paragraphs_embeddings,
                                  questions_embeddings,
                                  q_to_p,
                                  number_of_questions,
                                  outfile):
    neighbor_list = []
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        for _, neighbor_id in enumerate(neighbors):
            neighbor_list.append((_id,
                                  neighbor_id,
                                  (q_to_p[_id] == neighbor_id),
                                  True,
                                  sk_sim[neighbor_id],
                                  _,
                                  ))

    # -------------------------------------------------------------------------------------------------------
    # version 1 -> the following version could be useful for the AP handled in jupyter file in backup folders
    # this version needs to use that jupyter file to calculate recall@k as well.
    # -------------------------------------------------------------------------------------------------------
    # neighbor_list.append((_id,
    #                                   neighbor_id,
    #                                   _ + 1,
    #                                   sk_sim[neighbor_id],
    #                                   q_to_p[_id],
    #                                   np.where(neighbors == q_to_p[_id])[0][0] + 1,
    #                                   sk_sim[q_to_p[_id]]
    #                                   ))
    # df_neighbors = pd.DataFrame(data=neighbor_list, columns=['question',
    #                                                          'neighbor_paragraph',
    #                                                          'neighbor_order',
    #                                                          'neighbor_cos_similarity',
    #                                                          'actual_paragraph',
    #                                                          'actual_paragraph_order',
    #                                                          'actual_paragrraph_cos_similarity'
    #                                                          ])
    # df_neighbors.to_csv(outfile, index=False)
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------
    # verion 2 -> it is calculating recall@k on the fly, it still requires jupyter file to calculate AP,
    # -------------------------------------------------------------------------------------------------------
    columns = ['question', 'paragraph', 'ground_truth', 'is_model_answered_correctly',
               'cosine_score', 'nearest_neighbor_order']
    df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list, columns=columns)
    df_neighbor_within_paragraph = df_neighbor_within_paragraph[
        df_neighbor_within_paragraph['is_model_answered_correctly'] == True]

    df_neighbor_within_paragraph.to_csv(outfile.replace('###', ''), index=False)
    recall_ns = [1, 2, 5, 10, 20, 50]
    recall_columns = ['n', 'number_of_true', 'normalized_recalls']
    df_neighbor_within_paragraph_recalls = pd.DataFrame(data=calculate_recall_at_n(recall_ns,
                                                                                   df_neighbor_within_paragraph,
                                                                                   number_of_questions)
                                                        , columns=recall_columns
                                                        )

    df_neighbor_within_paragraph_recalls.to_csv(outfile.replace('###', 'recalls'),
                                                index=False)

#for each question, paragraph index is added to question to paragraph
def filter_and_calculate_similarity_and_dump(paragraphs_embeddings,
                                             questions_embeddings,
                                             paragraphs,
                                             data_eval,
                                             q_to_p,
                                             number_of_questions,
                                             outfile):
    neighbor_list = []
    questions_answers_not_found= dict()
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        ques_answers = data_eval[str(_id+1)]['answers']
        ques_id = data_eval[str(_id+1)]['uuid']
        _fo = 0
        for _, neighbor_id in enumerate(neighbors):
            if any_match_for_paragraph_and_answer( paragraphs[neighbor_id], ques_answers):
                neighbor_list.append((_id,
                                      neighbor_id,
                                      (q_to_p[_id] == neighbor_id),
                                      True,
                                      sk_sim[neighbor_id],
                                      _fo + 1
                                      ))
                _fo +=1

    # -------------------------------------------------------------------------------------------------------
    # version 1 -> the following version could be useful for the AP handled in jupyter file in backup folders
    # this version needs to use that jupyter file to calculate recall@k as well.
    # -------------------------------------------------------------------------------------------------------
    #  neighbor_list.append((slice_type,
    #                                       _id,
    #                                       neighbor_id,
    #                                       _fo + 1,
    #                                       #_ + 1,
    #                                       sk_sim[neighbor_id],
    #                                       q_to_p[_id],
    #                                       np.where(neighbors == q_to_p[_id])[0][0] + 1,
    #                                       sk_sim[q_to_p[_id]]
    #                                       ))
    # df_neighbors = pd.DataFrame(data=neighbor_list, columns=['slice_type',
    #                                                          'question',
    #                                                          'neighbor_paragraph',
    #                                                          'neighbor_order',
    #                                                          'neighbor_cos_similarity',
    #                                                          'actual_paragraph',
    #                                                          'actual_paragraph_order',
    #                                                          'actual_paragrraph_cos_similarity'
    #                                                          ])
    # df_neighbors.to_csv(outfile, index=False)

        # -------------------------------------------------------------------------------------------------------
        # verion 2 -> it is calculating recall@k on the fly, it still requires jupyter file to calculate AP,
        # -------------------------------------------------------------------------------------------------------
        columns = ['question', 'paragraph', 'ground_truth', 'is_model_answered_correctly',
                   'cosine_score', 'nearest_neighbor_order']
        df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list, columns=columns)
        df_neighbor_within_paragraph = df_neighbor_within_paragraph[
            df_neighbor_within_paragraph['is_model_answered_correctly'] == True]

        df_neighbor_within_paragraph.to_csv(outfile.replace('###', ''), index=False)
        recall_ns = [1, 2, 5, 10, 20, 50]
        recall_columns = ['n', 'number_of_true', 'normalized_recalls']
        df_neighbor_within_paragraph_recalls = pd.DataFrame(data=calculate_recall_at_n(recall_ns,
                                                                                       df_neighbor_within_paragraph,
                                                                                       number_of_questions)
                                                            , columns=recall_columns
                                                            )

        df_neighbor_within_paragraph_recalls.to_csv(outfile.replace('###', 'recalls'),
                                                    index=False)

def filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings,
                                                        questions_embeddings,
                                                        predictions,
                                                        paragraphs,
                                                        data_eval,
                                                        q_to_p,
                                                        number_of_questions,
                                                        outfile):
    neighbor_list_within_paragraph = []
    neighbor_list_match_answers = []
    paragraphs = [normalize_answer(p) for p in paragraphs]
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        ques_answers = data_eval[str(_id+1)]['answers']
        ques_id = data_eval[str(_id+1)]['uuid']
        pred_answer = normalize_answer(predictions[ques_id])
        #p_id = q_to_p[_id]
        #neighbor_id = neighbors[p_id]

        paragraphs_that_shared_answers = [p_id for p_id, p in enumerate(paragraphs) if pred_answer in p] #[0,1,2]

        # paragraph_index - decreasing order for cos (neighbor_id)
        # paragraph_order based on decreasing order for cos (_)
        # _id = question id
        # (q_to_p[_id] == neighbor_id) is it true paragraph
        for neighbor_id in paragraphs_that_shared_answers:
            _ = np.where(neighbors == neighbor_id)[0][0] + 1
            neighbor_list_within_paragraph.append(( _id,
                                                    neighbor_id,
                                                    (q_to_p[_id] == neighbor_id),
                                                    True,
                                                    sk_sim[neighbor_id],
                                                    _,
                                                    ))

        # for _, neighbor_id in enumerate(neighbors): #[2,5,7]
        #     is_answered_correctly = False
        #     [(i, s) for i, s in enumerate(paragraphs) if normalize_answer(pred_answer) in normalize_answer(s)]
        #     if any_match_for_paragraph_and_answer(paragraphs[neighbor_id], [pred_answer]):
        #         is_answered_correctly = True
        #         neighbor_list_within_paragraph.append(( _id,
        #                               neighbor_id,
        #                               (q_to_p[_id] == neighbor_id),
        #                               is_answered_correctly,
        #                               sk_sim[neighbor_id],
        #                               _ + 1,
        #                               ))
            # if is_answered_correctly:
            #     break
            # is_answered_correctly = False
            # if any_match_for_answer_and_prediction(pred_answer, ques_answers):
            #     is_answered_correctly = True
            # neighbor_list_match_answers.append((slice_type,
            #                   _id,
            #                   neighbor_id,
            #                   (q_to_p[_id] == neighbor_id),
            #                   is_answered_correctly,
            #                   sk_sim[neighbor_id],
            #                   _ + 1,
            #                   ))

    columns = ['question', 'paragraph', 'ground_truth', 'is_model_answered_correctly',
                'cosine_score', 'nearest_neighbor_order']

    df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list_within_paragraph, columns=columns)
    #df_neighbor_match_answers= pd.DataFrame(data=neighbor_list_match_answers, columns=columns)

    df_neighbor_within_paragraph = df_neighbor_within_paragraph[df_neighbor_within_paragraph['is_model_answered_correctly'] == True]
    # df_neighbor_match_answers = df_neighbor_match_answers[
    #     df_neighbor_match_answers['is_model_answered_correctly'] == True]

    df_neighbor_within_paragraph.to_csv(outfile.replace('###',''),index=False)
    #df_neighbor_match_answers.to_csv(outfile.replace('###','answer_matches_answer'), index=False)

    recall_ns = [1,2,5,10,20,50]
    recall_columns = ['n', 'number_of_true', 'normalized_recalls']
    df_neighbor_within_paragraph_recalls = pd.DataFrame(data = calculate_recall_at_n(recall_ns,
                                                                                     df_neighbor_within_paragraph,
                                                                                     number_of_questions)
                                                        , columns=recall_columns
                                                        )

    # df_neighbor_match_answers_recalls = pd.DataFrame(data=calculate_recall_at_n(recall_ns,
    #                                                                             df_neighbor_match_answers,
    #                                                                             number_of_questions)
    #                                                     , columns=recall_columns
    #                                                     )

    df_neighbor_within_paragraph_recalls.to_csv(outfile.replace('###', 'recalls'), index=False)
    # df_neighbor_match_answers_recalls.to_csv(outfile.replace('###', 'answer_matches_answer_recalls'), index=False)

def calculate_recall_at_n(ns, data, number_of_questions):
    recalls = []
    for i in ns:
        total_number = len(data[(data['nearest_neighbor_order'] <= i) & (data['ground_truth'] == True) ])
        recalls.append((i, total_number, total_number/number_of_questions))
    return recalls

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
                        token_embeddings_file, voc_file_name, partition=20, weight_file = None, options_file = None,
                        is_google_elmo = False):

    # if os.path.exists(token_embeddings_file.replace('@@', 'new_api_with_old'))\
    #         or os.path.exists(token_embeddings_file.replace('@@', 'new_api_with_old')):
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
            t = token.strip()
            if t == "" or t is None:
                #print("Token: {}, Doc Indx: {}, Doc: {}".format(token, i, sentence))
                t = '-'
            corpus_as_tokens.append(t)


    UTIL.save_as_pickle(document_embedding_guideline, token_embeddings_guideline_file)

    all_tokens = set(['<S>', '</S>','<UNK>'])
    for token in corpus_as_tokens:
        all_tokens.add(token)
    with open(voc_file_name, 'w') as fout:
        fout.write('\n'.join(all_tokens))

    corpus_as_tokens = corpus_as_tokens if resource['is_non_context'] else tokenized_questions + tokenized_paragraphs
    corpus_as_tokens = corpus_as_tokens if resource['is_demo_slice'] is None else corpus_as_tokens[0:resource['is_demo_slice']]
    total_parition_size_for_old_api = math.ceil(len(corpus_as_tokens) / resource['partition_size'])
    if not is_google_elmo:
        method = 'only_old'
        document_embeddings = None
        if os.path.exists(token_embeddings_file.replace('@@', str('old_api_@@')).replace('@@', str(total_parition_size_for_old_api))):
            print('ELMO embeddings are found from the old API and loaded')
            for partition_index in range(1, total_parition_size_for_old_api + 1):
                with h5py.File(token_embeddings_file.replace('@@', str('old_api_@@')).replace('@@', str(partition_index)), 'r') as fin:
                    print(partition_index)
                    embedding = fin['embeddings'][...]
                    if document_embeddings is None:
                        document_embeddings = embedding
                    else:
                        document_embeddings = np.vstack((document_embeddings, embedding))
        else:
            print('ELMO embeddings are generated by using old API')
            document_embeddings= UTIL.generate_and_dump_elmo_embeddings(corpus_as_tokens,
                                                                     resource['is_non_context'],
                                                                     voc_file_name,
                                                                     elmo_dataset_file,
                                                                     elmo_options_file,
                                                                     weight_file,
                                                                     token_embeddings_file.replace('@@', str(
                                                                         'old_api_@@')),
                                                                     resource['partition_size']
                                                                     )
    else:
        method = 'new_api_with_old'
        if not os.path.exists(token_embeddings_file.replace('@@', 'new_api')):
            """
            ******************************************************************************************************************
            ******************************************************************************************************************
            START: GOOGLE ELMO EMBEDDINGS
            ******************************************************************************************************************
            ******************************************************************************************************************
            """
            print('ELMO embeddings are getting generated by Google ELMo')
            document_embeddings = None
            documents = corpus_as_tokens
            begin_index = 0
            reset_every_iter = 3
            batch = resource['batch']
            counter = 0
            while begin_index <= len(documents)-1:
                if counter % reset_every_iter == 0:
                    print('Graph is resetted')
                    tf.reset_default_graph()
                    elmo_embed = load_module("https://tfhub.dev/google/elmo/2", trainable=True)
                    tf.logging.set_verbosity(tf.logging.ERROR)

                begin_index = begin_index
                end_index = begin_index + batch
                if end_index > len(documents):
                    end_index = len(documents)
                print('Processing {} from {} to {}'.format('all documents', begin_index, end_index))
                with tf.Session() as session:
                    print('Session is opened')
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

                    d1 = session.run(elmo_embed(documents[begin_index:end_index],
                                                signature="default",
                                                as_dict=True)['lstm_outputs1'])

                    d2 = session.run(elmo_embed(documents[begin_index:end_index],
                                                signature="default",
                                                as_dict=True)['lstm_outputs2'])

                    delmo = session.run(elmo_embed(documents[begin_index:end_index],
                                                   signature="default",
                                                   as_dict=True)['elmo'])
                    # for i, each_document in enumerate(tqdm(tokenized[begin_index:end_index],
                    #                                        total=len(tokenized[begin_index:end_index])), begin_index):

                    stacked_embeddings = []
                    for doc_index, embed_document in enumerate(enumerate(documents[begin_index:end_index]),
                                                                   begin_index):

                        embed_index, each_document = embed_document
                        _begining = 0
                        _ending = 1
                        _d1 = d1[embed_index, _begining:_ending, :]
                        #UTIL.dump_embeddings(_d1, embedding_file.replace('@', 'LSTM1_' + str(doc_index)))
                        _d2 = d2[embed_index, _begining:_ending, :]
                        #UTIL.dump_embeddings(_d2, embedding_file.replace('@', 'LSTM2_' + str(doc_index)))
                        _delmo = delmo[embed_index, _begining:_ending, :]
                        #UTIL.dump_embeddings(_delmo, embedding_file.replace('@', 'ELMO_' + str(doc_index)))
                        stacked_embedding = np.vstack([_delmo, _d1, _d2])
                        stacked_embeddings.append(stacked_embedding)

                    stacked_embeddings = np.asarray(stacked_embeddings)
                    if document_embeddings is None:
                        document_embeddings = stacked_embeddings
                    else:
                        document_embeddings = np.vstack((document_embeddings, stacked_embeddings))
                    #document_embeddings.append(stacked_embeddings)
                    counter += 1
                    begin_index += batch
            UTIL.dump_embeddings(document_embeddings, token_embeddings_file.replace('@@', 'new_api'))
        else:
            print('ELMO embeddings are found and loaded')
            document_embeddings = UTIL.load_embeddings(token_embeddings_file.replace('@@', 'new_api'))
            """
            ******************************************************************************************************************
            ******************************************************************************************************************
            END: GOOGLE ELMO EMBEDDINGS
            ******************************************************************************************************************
            ******************************************************************************************************************
            """
        ####################################################################################
        ### OLD API #######################################################################
        ####################################################################################
        document_embeddings_old = None
        if os.path.exists(token_embeddings_file.replace('@@', str('old_api_@@')).replace('@@', str(total_parition_size_for_old_api))):
            print('ELMO embeddings are found from the old API and loaded')
            for partition_index in range(1, total_parition_size_for_old_api + 1):
                with h5py.File(token_embeddings_file.replace('@@', str('old_api_@@')).replace('@@', str(partition_index)), 'r') as fin:
                    print(partition_index)
                    embedding = fin['embeddings'][...]
                    if document_embeddings_old is None:
                        document_embeddings_old = embedding
                    else:
                        document_embeddings_old = np.vstack((document_embeddings_old, embedding))
        else:
            print('ELMO embeddings are generated by using old API')
            document_embeddings_old = UTIL.generate_and_dump_elmo_embeddings(corpus_as_tokens,
                                                                     resource['is_non_context'],
                                                                     voc_file_name,
                                                                     elmo_dataset_file,
                                                                     elmo_options_file,
                                                                     weight_file,
                                                                     token_embeddings_file.replace('@@', str(
                                                                         'old_api_@@')),
                                                                     resource['partition_size']
                                                                     )
        # INSERT OLD ONES BEST VALUES TO NEW ONES SO THAT WE HAVE MORE LAYERS
        print('OLD and NEW Embeddings are inserted')
        document_embeddings = np.insert(document_embeddings, 3, document_embeddings_old, axis=1)
    print('Embeddings are being written')
    UTIL.dump_embeddings(document_embeddings, token_embeddings_file.replace('@@', method))
    # else:
    #     document_embedding_guideline = UTIL.load_from_pickle(token_embeddings_guideline_file)
    #     document_embeddings = UTIL.load_embeddings(token_embeddings_file.replace('@@', method))

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
def load_embeddings(infile_to_get):
    with h5py.File(infile_to_get, 'r') as fin:
        document_embeddings = fin['embeddings'][...]
    return document_embeddings
def create_idf_matrix(tokenized_questions, tokenized_paragraphs, token2idfweight):
    idf_matrix = []
    for sentence in tokenized_questions + tokenized_paragraphs:
        for word in sentence:
            idf_matrix.append(token2idfweight[word])

    idf_matrix = np.asarray(idf_matrix)
    idf_matrix = idf_matrix.reshape(idf_matrix.shape[0], 1,1)
    return idf_matrix

def token_to_document_embeddings(tokenized_questions, tokenized_paragraphs,token_embeddings, token_embeddings_guideline, weight_matrix):
    questions_embeddings = []
    paragraphs_embeddings = []
    documents = tokenized_questions + tokenized_paragraphs
    for _ in tqdm(range(len(documents))):
        str_index = token_embeddings_guideline[_]['start_index']
        end_index = token_embeddings_guideline[_]['end_index']
        d_type = token_embeddings_guideline[_]['type']
        if d_type == 'q':
            questions_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_question_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))
        else:
            paragraphs_embeddings.append(np.mean(token_embeddings[str_index:end_index, :, :], axis=0))
            # idf_paragraph_matrix.append(np.mean(idf_vec[str_index:end_index], axis=0))

    #     # -------------------------------------------------
    #     #after validation, this is going to be deleted
    #     # -------------------------------------------------
    #     simplified_token_embeddings = {}
    #     if is_inhouse_elmo_for_cnn:
    #         # simplified token embeddings for CNN module
    #         for token_index, token_name in enumerate(documents[_]):
    #             _tok_str_index = str_index + token_index
    #             _tok_end_index = _tok_str_index + 1
    #             simplified_token_embedding = np.mean(np.multiply(np.asarray(token_embeddings[_tok_str_index:_tok_end_index, :, :]), weight_matrix),axis=1)
    #             saved_token_embeddings = simplified_token_embeddings.get(token_name)
    #             if saved_token_embeddings is not None:
    #                 if saved_token_embeddings == simplified_token_embedding:
    #                     print('{} have always same embeddings'.format(token_name))
    #             else:
    #                 simplified_token_embeddings[token_name] = simplified_token_embedding
    #
    #     token_to_idx = {}
    #     simplified_token_embeddings = []
    #     if is_inhouse_elmo_for_cnn:
    #         # simplified token embeddings for CNN module
    #         glboal_token_index = 0
    #         for token_index, token_name in enumerate(documents[_]):
    #             _tok_str_index = str_index + token_index
    #             _tok_end_index = _tok_str_index + 1
    #             simplified_token_embedding_ = np.mean(
    #                 np.multiply(np.asarray(token_embeddings[_tok_str_index:_tok_end_index, :, :]), weight_matrix),
    #                 axis=1)
    #             simplified_token_embedding = np.asarray(token_embeddings[_tok_str_index:_tok_end_index, :, :])
    #             saved_token_embeddings = token_to_idx.get(token_name)
    #             if saved_token_embeddings is None:
    #                 token_to_idx[token_name] = glboal_token_index
    #                 simplified_token_embeddings.append(simplified_token_embedding)
    #                 glboal_token_index +=1
    #         simplified_token_embeddings = np.asarray(simplified_token_embeddings)
    #         simplified_token_embeddings = np.multiply(simplified_token_embeddings, weight_matrix)
    #         simplified_token_embeddings = np.mean(simplified_token_embeddings, axis=1)
    #
    #         vocabulary_size = len(token_to_idx)
    #         # Assume your embeddings variable
    #         tok_embeddings = tf.Variable(simplified_token_embedding)
    #         with tf.Session() as sess:
    #             embeddings_val = sess.run(tok_embeddings)
    #             with open(token_embeddings_file.replace('@@', 'CNN').replace('hdf5','txt'), 'w') as file_:
    #                 for i in range(vocabulary_size):
    #                     embed = embeddings_val[i, :]
    #                     word = token_to_idx[i]
    #                     file_.write('%s %s\n' % (word, ' '.join(map(str, embed))))
    # del token_embeddings
    # del simplified_token_embeddings

    questions_embeddings = np.asarray(questions_embeddings)
    paragraphs_embeddings = np.asarray(paragraphs_embeddings)

    questions_embeddings = np.multiply(questions_embeddings, weight_matrix)
    paragraphs_embeddings = np.multiply(paragraphs_embeddings, weight_matrix)

    questions_embeddings = np.mean(questions_embeddings, axis=1)
    paragraphs_embeddings = np.mean(paragraphs_embeddings, axis=1)

    return questions_embeddings, paragraphs_embeddings

def transform_to_idf_weigths(tokenized_questions, tokenized_paragraphs, tokenizer, questions_nontokenized,paragraphs_nontokenized):
    tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenizer)
    tfidf.fit(questions_nontokenized + paragraphs_nontokenized)
    max_idf = max(tfidf.idf_)
    token2idfweight = defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    idf_vec = create_idf_matrix(tokenized_questions, tokenized_paragraphs, token2idfweight)
    return token2idfweight, idf_vec

def transorm_to_glove_embeddings(local_vector_model, token2idfweight, dims, tokenized_questions, tokenized_paragraphs, is_dump, glove_question_embeddings_file, glove_paragraph_embeddings_file):
    mean_glove_with_idf_question_embeddings = np.array([
        np.mean([local_vector_model.word_vectors[local_vector_model.dictionary[w]] * token2idfweight[w]
                 for w in words if w in local_vector_model.dictionary] or
                [np.zeros(dims)], axis=0)
        for words in tokenized_questions
    ])
    mean_glove_with_idf_paragraph_embeddings = np.array([
        np.mean([local_vector_model.word_vectors[local_vector_model.dictionary[w]] * token2idfweight[w]
                 for w in words if w in local_vector_model.dictionary] or
                [np.zeros(dims)], axis=0)
        for words in tokenized_paragraphs
    ])
    if is_dump:
        dump_embeddings(mean_glove_with_idf_question_embeddings, glove_question_embeddings_file)
        dump_embeddings(mean_glove_with_idf_paragraph_embeddings, glove_paragraph_embeddings_file)
    return mean_glove_with_idf_question_embeddings, mean_glove_with_idf_paragraph_embeddings

def dump_mapping_data(data, outfile_to_dump):
    data_df = pd.DataFrame(np.array(data), columns=list("v"))
    data_df.to_csv(outfile_to_dump)

def load_module(module_url = "https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False):
    if trainable:
        embed = hub.Module(module_url, trainable)
    else:
        embed = hub.Module(module_url)
    return embed

TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = DEV

laptop={"batch": 500,
        "partition_size":500,
         "is_non_context":True,
         "is_demo_slice":1003,
        }

titanX={"batch": 5000,
        "partition_size":50000,
         "is_non_context":False,
        "is_demo_slice":None
        }
resource=titanX

is_dump_during_execution = True
is_inject_idf = False
is_filtered_by_answers_from_rnet = False

# SPLIT DOCS
is_split_content_to_documents = False
split_num_of_paragrahs_in_slices = 1000
percent_of_slice_splits = .4

# ELMO EMBEDDINGS #
is_elmo_embeddings= True
is_elmo_document_embeddings_already_generated = False
partition_size = 1
is_elmo_word_embeddings_already_generated = False
is_calculate_recalls = False

# USE EMBEDDINGS #
is_use_embedding = False

# GOOGLE ELMO EMBEDDINGS #
is_google_elmo_embedding = False

# CREATE 3D INHOUSE ELMO FOR CNN
is_inhouse_elmo_for_cnn = False

# IMPROVE ELMEDDINGS
is_improve_elmeddings=False

# # GLOVE TRAINING #
# is_inject_local_weights = False
# is_force_to_train_local_corpus = False
# glove_window = 10
# glove_dims = 1024
# glove_learning_rate = 0.05
# glove_epoch = 300
# glove_threads = 10
# local_embedding_models = ['glove']
################ CONFIGURATIONS #################


_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_paragraphs_file_name = '{}_paragraphs.txt'
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embeddings.hdf5'.format(dataset_type)
paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

_token_embeddings_file_name = '{}_token_embeddings_@@.hdf5'.format(dataset_type)
token_embeddings_file = os.path.join(datadir, '.',_token_embeddings_file_name)

_token_embeddings_guideline_file_name = '{}_token_embeddings_guideline.pkl'.format(dataset_type)
token_embeddings_guideline_file = os.path.join(datadir, _token_embeddings_guideline_file_name)

_questions_file_name = '{}_questions.txt'
questions_file = os.path.join(datadir, _questions_file_name)

_mapping_file_name = '{}_q_to_p_mappings.csv'
mapping_file = os.path.join(datadir, _mapping_file_name)

_question_embeddings_file_name = '{}_question_embeddings.hdf5'.format(dataset_type)
question_embeddings_file = os.path.join(datadir, _question_embeddings_file_name)

_word_embeddings_file_name = '{}_word_embeddings.hdf5'.format(dataset_type)
word_embeddings_file = os.path.join(datadir, _word_embeddings_file_name)

_neighbors_file_name = '{}_neighbors.csv'.format(dataset_type)
neighbors_file = os.path.join(datadir, _neighbors_file_name)

_voc_file_name = '{}_voc.txt'.format(dataset_type)
voc_file_name = os.path.join(datadir, _voc_file_name)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

_glove_file_name = 'glove.840B.300d.txt'
glove_file = os.path.join(datadir, _glove_file_name)

answers_file = os.path.join(datadir, '{}_answer.json'.format(dataset_type))

file_pattern_for_local_weights = os.path.join(datadir,"{}_local_weights_{}_embeddings.hdf5")
file_pattern_for_local_model = os.path.join(datadir,"{}_local_{}.model")
glove_local_question_embeddings_file= os.path.join(datadir, "{}_glove_local_question_embeddings.hdf5".format(dataset_type))
glove_local_paragraph_embeddings_file= os.path.join(datadir, "{}_glove_local_paragraph_embeddings.hdf5".format(dataset_type))

split_prefix = '{}_paragraphs'.format(dataset_type)

elmo_weights_file = os.path.join(datadir, 'weights.hdf5')
elmo_options_file = os.path.join(datadir, 'options.json')
elmo_dataset_file = os.path.join(datadir, '{}_dataset_file.txt'.format(dataset_type))

print('Squad Data: Processing Started')
start = datetime.datetime.now()
# paragraphs, questions, q_to_p = read_squad_data(squad_other_file)
# paragraphs_test, questions_test, q_to_p_test = read_squad_data(squad_file)
train_word_counter, train_char_counter, dev_word_counter, dev_char_counter = Counter(), Counter(), Counter(), Counter()
dev_examples, dev_eval, dev_questions, dev_paragraphs, dev_q_to_ps = process_file(squad_file.format(DEV), DEV,
                                                                                  dev_word_counter, dev_char_counter)

print('# of Paragraphs in Dev : {}'.format(len(dev_paragraphs)))
print('# of Questions in Dev: {}'.format(len(dev_questions)))
print('# of Q_to_P Dev: {}'.format(len(dev_q_to_ps)))

print(20 * '-')
print('Paragraphs: Tokenization and Saving Tokenization Started in Dev')
tokenized_paragraphs = tokenize_contexts(dev_paragraphs)
print('# of Tokenized Paragraphs in Dev: {}'.format(len(tokenized_paragraphs)))
print(20 * '-')
print('Questions: Tokenization and Saving Tokenization Started')
tokenized_questions = tokenize_contexts(dev_questions)
print('# of Tokenized Questions in Dev: {}'.format(len(tokenized_questions)))

questions_nontokenized = [" ".join(context) for context in tokenized_questions]
paragraphs_nontokenized = [" ".join(context) for context in tokenized_paragraphs]

if is_dump_during_execution:
    dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file.format(DEV))
    dump_tokenized_contexts(tokenized_questions, questions_file.format(DEV))
    dump_mapping_data(dev_q_to_ps, mapping_file.format(DEV))

examples = dev_examples
eval = dev_eval
questions = dev_questions
paragraphs = dev_paragraphs
q_to_ps = dev_q_to_ps
if dataset_type == TRAIN:
    train_examples, train_eval, train_questions, train_paragraphs, train_q_to_ps = process_file(
        squad_file.format(TRAIN), TRAIN, train_word_counter, train_char_counter)
    print('#' * 20)
    print('# of Paragraphs in Train : {}'.format(len(train_paragraphs)))
    print('# of Questions in Train: {}'.format(len(train_questions)))
    print('# of Q_to_P in Train: {}'.format(len(train_q_to_ps)))

    print(20 * '-')
    print('Paragraphs: Tokenization and Saving Tokenization Started in Train')
    tokenized_train_paragraphs = tokenize_contexts(train_paragraphs)
    print('# of Tokenized Paragraphs in Train: {}'.format(len(tokenized_train_paragraphs)))
    print(20 * '-')
    print('Questions: Tokenization and Saving Tokenization Started')
    tokenized_train_questions = tokenize_contexts(train_questions)
    print('# of Tokenized Questions in Train: {}'.format(len(tokenized_train_questions)))

    questions_nontokenized = [" ".join(context) for context in tokenized_train_questions]
    paragraphs_nontokenized = [" ".join(context) for context in tokenized_train_paragraphs]

    if is_dump_during_execution:
        dump_tokenized_contexts(tokenized_train_paragraphs, paragraphs_file.format(TRAIN))
        dump_tokenized_contexts(tokenized_train_questions, questions_file.format(TRAIN))
        dump_mapping_data(train_q_to_ps, mapping_file.format(TRAIN))

    tokenized_questions = tokenized_train_questions
    tokenized_paragraphs = tokenized_train_paragraphs
    examples = train_examples
    eval = train_eval
    questions = train_questions
    paragraphs = train_paragraphs
    q_to_ps = train_q_to_ps
end = datetime.datetime.now()
print('Squad Data: Processing Ended in {} minutes'.format((end - start).seconds / 60))


if is_split_content_to_documents:
    print('Contents are getting splitted to documents')
    start = datetime.datetime.now()
    contexts_vocs = set()
    total_tokens = 0
    if not os.path.exists(os.path.join(datadir, split_prefix)):
        os.makedirs(os.path.join(datadir, split_prefix))
    if not os.path.exists(os.path.join(datadir, split_prefix, 'test')):
        os.makedirs(os.path.join(datadir, split_prefix,'test'))

    i, slice_start, num_of_paragrahs_in_slices = 0, 0, split_num_of_paragrahs_in_slices
    num_of_all_slice = int(len(tokenized_paragraphs) / num_of_paragrahs_in_slices)
    num_of_test_slices = int(num_of_all_slice * percent_of_slice_splits)
    test_slices = random.sample(range(0, num_of_all_slice), num_of_test_slices)
    while i <= num_of_all_slice:
        slice_start = i * num_of_paragrahs_in_slices
        context = list(chain.from_iterable(tokenized_paragraphs[slice_start:slice_start + num_of_paragrahs_in_slices]))
        i+=1
        segmented_context = sentence_segmenter(" ".join(context))
        segmented_context = [span.text for span in segmented_context]
        if i not in test_slices:
            [contexts_vocs.add(__) for __ in context if __.strip()]
            total_tokens += len(context)
            with open(os.path.join(datadir, split_prefix, str(i)+'.txt'), 'w') as fout:
               fout.write('\n'.join(segmented_context))
        else:
            with open(os.path.join(datadir, split_prefix, 'test', str(i)+'.txt'), 'w') as fout:
               fout.write('\n'.join(segmented_context))

    mandatory_tokens_written = False
    with open(os.path.join(datadir, '{}_contexts_voc.txt'.format(dataset_type)), 'w') as f_vout:
        if not mandatory_tokens_written:
            f_vout.write('\n'.join(['<S>', '</S>', '<UNK>']) + '\n')
            mandatory_tokens_written = True
        f_vout.write('\n'.join(contexts_vocs))
    end = datetime.datetime.now()
    print('Number of tokens in corpus: {}'.format(total_tokens))
    print('Number of unique tokens in corpus: {}'.format(len(contexts_vocs)))
    print('Splitting is completed in {} minutes'.format((end-start).seconds/60))

if is_elmo_embeddings or is_google_elmo_embedding:
    if not is_elmo_document_embeddings_already_generated:
        if not is_elmo_word_embeddings_already_generated:
            print('\n')
            print(20* '-')
            print('ELMO Token Embeddings is started')
            start = datetime.datetime.now()
            #########################
            # PARTITION AND MERGE THE FILES AND WORDS AS SENTECES
            #########################
            # total_tokens_in_each_embedding_file = 715372
            #
            # par_str_doc_first_index = len(tokenized_train_questions)
            # par_str_doc_last_index = len(tokenized_train_questions + tokenized_train_paragraphs) - 1
            #
            # par_token_str_index = document_embedding_guideline[par_str_doc_first_index]['start_index']
            # par_token_end_index = document_embedding_guideline[par_str_doc_last_index]['end_index']
            #
            # partitioned_embs_files_start_indx = math.ceil(par_token_str_index/total_tokens_in_each_embedding_file)
            # partitioned_embs_files_end_indx = math.ceil(par_token_end_index/total_tokens_in_each_embedding_file)
            # is_first_record = True
            #for partition_index in range(partitioned_embs_files_start_indx, partitioned_embs_files_end_indx + 1):
            token_embeddings, document_embedding_guideline = get_elmo_embeddings(tokenized_questions,
                                                                               tokenized_paragraphs,
                                                                               token_embeddings_guideline_file,
                                                                               token_embeddings_file,
                                                                               voc_file_name,
                                                                               partition_size,
                                                                               weight_file=elmo_weights_file,
                                                                               options_file=elmo_options_file,
                                                                               is_google_elmo=is_google_elmo_embedding
                                                                               )
            end = datetime.datetime.now()
            print('ELMO Token Embeddings is ended in {} minutes'.format((end-start).seconds/60))





            print('\n')
            print(20* '-')
            if is_inject_idf:
                print('IDF is going to be calculated')
                _type = 'with_idf'
                start = datetime.datetime.now()
                token2idfweight, idf_vec = transform_to_idf_weigths(tokenized_questions, tokenized_paragraphs, tokenize, questions_nontokenized, paragraphs_nontokenized)
                weighted_token_embeddings = np.multiply(idf_vec, token_embeddings)
                end = datetime.datetime.now()
                print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
            else:
                print('IDF is skipped')
                _type = 'only'
                weighted_token_embeddings = token_embeddings







        print('\n')
        print(20 * '-')
        print('ELMO Embeddings is started')
        start = datetime.datetime.now()
        _a_b_c_d_s = []
        # _a_b_c_s.append([0,0,1])
        # _a_b_c_s.append([0, 1, 0])
        _a_b_c_d_s.append([1, 0, 0, 0])
        _a_b_c_d_s.append([0, 1, 0, 0])
        _a_b_c_d_s.append([0, 0, 1, 0])
        _a_b_c_d_s.append([0, 0, 0, 1])
        # while len(_a_b_c_s) < 10:
        #     x = np.random.dirichlet(np.ones(3), size=1).tolist()
        #     x_ = [float("{:1.2f}".format(_x)) for _x in x[0]]
        #     total_x_ = sum(x_)
        #     if total_x_ == 1:
        #         _a_b_c_s.append(x_)
        #         _a_b_c_s = sort_and_deduplicate(_a_b_c_s)

        # for _token_embed_pack in [(weighted_token_embeddings, 'with_idf'), (token_embeddings, 'only')]:


        _token_embed = weighted_token_embeddings if not is_elmo_word_embeddings_already_generated else None
        for _a_b_c_d in _a_b_c_d_s:
            print('Weight {}'.format(_a_b_c_d))

            if not is_elmo_word_embeddings_already_generated:
                WM = np.array([_a_b_c_d[0], _a_b_c_d[1], _a_b_c_d[2]],_a_b_c_d[3]).reshape((1, 4, 1))
                questions_embeddings, paragraphs_embeddings = token_to_document_embeddings(tokenized_questions,
                                                                                           tokenized_paragraphs,
                                                                                           _token_embed,
                                                                                           document_embedding_guideline,
                                                                                            WM)

                # dump, embeddings
            else:
                with h5py.File(question_embeddings_file, 'r') as fq_in, h5py.File(paragraph_embeddings_file, 'r') as fp_in:
                    paragraphs_embeddings = fp_in['embeddings'][...]
                    questions_embeddings = fq_in['embeddings'][...]

            print('Nearest Neighbors: Starting')
            sub_start = datetime.datetime.now()

            if is_calculate_recalls:
                # -------------------------------------------#
                # option 1 [WITH ALL PARAGRAPHS] depends on your scenerio           #
                # -------------------------------------------#
                calculate_similarity_and_dump(paragraphs_embeddings,
                                              questions_embeddings,
                                              q_to_ps,
                                              len(questions),
                                              os.path.join(datadir,
                                                           'elmo_{}_weights_a_{}_b_{}_c_{}_d_{}_output_neighbors_###.csv'.format(_type,
                                                                                                                                 _a_b_c_d[0],
                                                                                                                                 _a_b_c_d[1],
                                                                                                                                 _a_b_c_d[2],
                                                                                                                                 _a_b_c_d[3])))

                # -------------------------------------------#
                # option 2 [FILTERED PARAGRAPHS BY COMMON KEYWORDS IN QUESTION] depends on your scenerio          #
                # -------------------------------------------#
                filter_and_calculate_similarity_and_dump(paragraphs_embeddings,
                                                         questions_embeddings,
                                                         paragraphs,
                                                         eval,
                                                         q_to_ps,
                                                         len(questions),
                                              os.path.join(datadir,
                                                           'elmo_{}_weights_a_{}_b_{}_c_{}_d_{}_output_filtered_by_paragraphs_neighbors_###.csv'.format(_type,
                                                                                                                        _a_b_c_d[0],
                                                                                                                        _a_b_c_d[1],
                                                                                                                        _a_b_c_d[2],
                                                                                                                        _a_b_c_d[3])))

                # -------------------------------------------#
                # option 3 [FILTERED PARAGRAPHS BY ANSWERS FROM RNET MODEL] depends on your scenerio          #
                # -------------------------------------------#
                if is_filtered_by_answers_from_rnet:
                    with open(answers_file) as prediction_file:
                        answers = json.load(prediction_file)


                    filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, answers,
                                                                        paragraphs, eval,
                                                                        q_to_ps, len(questions),
                                                                        os.path.join(datadir,
                                                                                     'elmo_{}_weights_a_{}_b_{}_c_{}_d_{}_output_filtered_rnet_answers_neighbors_###.csv'.format(
                                                                                         _type,
                                                                                         _a_b_c_d[0],
                                                                                         _a_b_c_d[1],
                                                                                         _a_b_c_d[2],
                                                                                         _a_b_c_d[3])))
                sub_end = datetime.datetime.now()
                print('Nearest Neighbors: Completed is completed in {} minutes'.format((sub_end - sub_start).seconds / 60))

        questions_elmo_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[1]))
        dump_embeddings(questions_elmo_embeddings, question_embeddings_file)
        paragraphs_elmo_embeddings = np.reshape(paragraphs_embeddings,
                                           (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
        dump_embeddings(paragraphs_elmo_embeddings, paragraph_embeddings_file)

    else:
        ## document embeddings already generated
        ## get these embeddings:
        print('Document embeddings are getting loaded...')
        paragraphs_elmo_embeddings = load_embeddings(paragraph_embeddings_file)
        questions_elmo_embeddings = load_embeddings(question_embeddings_file)
        print('Document embeddings are loaded...')
    end = datetime.datetime.now()
    print('ELMO Embeddings is completed in {} minutes'.format((end - start).seconds / 60))
    print(20 * '-')
elif is_use_embedding : # USE Embedding Process
    use_embed = load_module("https://tfhub.dev/google/universal-sentence-encoder/2")
    print("Question Len", len(questions_nontokenized))
    print("Paragraphs Len", len(paragraphs_nontokenized))
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        question_embeddings = session.run(use_embed(questions_nontokenized))
        paragraph_embeddings = session.run(use_embed(paragraphs_nontokenized))
        dump_embeddings(question_embeddings, question_embeddings_file + "_USE")
        dump_embeddings(paragraph_embeddings, paragraph_embeddings_file + "_USE")

    end = datetime.datetime.now()
    print('USE Embeddings is completed in {} minutes'.format((end - start).seconds / 60))
    print(20 * '-')

elif is_google_elmo_embedding:
    print("Question Len", len(questions_nontokenized))
    print("Paragraphs Len", len(paragraphs_nontokenized))
    question_len = [len(q) for q in tokenized_questions]
    max_question_len = max(question_len)
    padded_tokenized_question = [pad_list(q, max_question_len, '') for q in tokenized_questions]

    paragraph_len = [len(p) for p in tokenized_paragraphs]
    max_paragraph_len = max(paragraph_len)
    padded_tokenized_paragraph = [pad_list(p, max_paragraph_len, '') for p in tokenized_paragraphs]
    """
    SPECIAL PARAMETERS FOR GOOGLE ELMO
    """
    embedding_type = 'elmo'
    batch_size_for_q = 1000
    batch_size_for_p = 1 #math.ceil(len(paragraphs_nontokenized) / loop)
    save_for_every_batch = 50
    # 0 - 100 : 1
    # 100 - 200 : 2
    # 200 - 300 : 3
    # 300 - 400 : 4
    # 400 - 500 : 5
    # 500 - 600 : 6
    # 600 - 700 : 7
    # 700 - 800 : 8
    # 800 - 900 : 9

    batch_index_to_be_resumed = 1000
    counter_for_saving = 11

    # is it for saving embeddings or reading embeddings to have one embedding file?
    is_generating_embeddings = True
    # is it running for questions?
    question_step = False
    # is it running for paragraphs?
    paragrap_step = True


    if question_step:
        print('Questions are getting computed')
        total_number_of_questions = len(questions_nontokenized)
        print('Total Number of questions: {}'.format(total_number_of_questions))
        batch_counter = int(math.ceil(total_number_of_questions / batch_size_for_q))
        print('Total Number of batches: {}'.format(batch_counter))
        print('Batch Size: {}'.format(batch_size_for_q))
        if is_generating_embeddings:
            execution_counter = 0
            while True:
                tf.reset_default_graph()
                elmo_embed = load_module("https://tfhub.dev/google/elmo/2", trainable=True)
                tf.logging.set_verbosity(tf.logging.ERROR)
                execution_counter +=1
                print(50 * '=')
                print('Exceution {}'.format(execution_counter))
                print(50 * '=')
                with tf.Session() as session:
                    question_embeddings = None
                    timer_for_saving= save_for_every_batch
                    for i in range(1, batch_counter + 1):
                        if i * batch_size_for_q > total_number_of_questions:
                            print('There is no more questions left, batch counter is out of range {}/{}'.format(i * batch_size_for_q, total_number_of_questions))
                            if timer_for_saving != save_for_every_batch:
                                dump_embeddings(question_embeddings,
                                                question_embeddings_file + "_google_elmo_" + str(counter_for_saving))
                                print(question_embeddings_file + "_google_elmo_" + str(counter_for_saving), ' is dumped')
                                del question_embeddings
                                question_embeddings = None
                                counter_for_saving += 1
                                batch_index_to_be_resumed += save_for_every_batch
                                timer_for_saving = save_for_every_batch
                                exit()
                        else:
                            if batch_index_to_be_resumed == -1 or (batch_index_to_be_resumed != -1 and i > batch_index_to_be_resumed):
                                # if there is still enough questions for batch:
                                _begin_time = datetime.datetime.now()
                                print('Batch Index: {}'.format(i))
                                _begin_index = (i - 1) * batch_size_for_q
                                _end_index = _begin_index + batch_size_for_q
                                print('Question Begin Index: {}'.format(_begin_index))
                                print('Question End Index: {}'.format(_end_index))

                                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                                q_e = session.run(elmo_embed(
                                    inputs={
                                        "tokens": padded_tokenized_question[_begin_index:_end_index],
                                        "sequence_len": question_len[_begin_index:_end_index]
                                    },
                                    signature="tokens",
                                    as_dict=True)[embedding_type])

                                if question_embeddings is None:
                                    question_embeddings = np.mean(q_e, axis=1)
                                else:
                                    question_embeddings = np.vstack((question_embeddings, np.mean(q_e, axis=1)))

                                timer_for_saving -=1
                                if timer_for_saving == 0:
                                    dump_embeddings(question_embeddings, question_embeddings_file + "_google_elmo_" + str(counter_for_saving))
                                    print(question_embeddings_file + "_google_elmo_" + str(counter_for_saving), ' is dumped')
                                    del question_embeddings
                                    question_embeddings = None
                                    counter_for_saving +=1
                                    batch_index_to_be_resumed += save_for_every_batch
                                    timer_for_saving = save_for_every_batch
                                    break
                                _end_time = datetime.datetime.now()
                                print('Loop {} takes {} mins'.format(i, (_end_time - _begin_time).seconds/ 60))
                                print(50 * '+')
        else:
            all_question_embeddings = None
            for i in range(1, counter_for_saving+1):
                _all_question_embeddings = load_embeddings(question_embeddings_file + "_google_elmo_" + str(i))
                if all_question_embeddings is None:
                    all_question_embeddings = _all_question_embeddings
                else:
                    all_question_embeddings = np.vstack((all_question_embeddings, _all_question_embeddings))

            dump_embeddings(all_question_embeddings, question_embeddings_file + "_google_elmo")
            print(question_embeddings_file + "_google_elmo", ' is dumped')

    if paragrap_step:
        print('Paragraphs are getting computed')
        total_number_of_paragraphs = len(paragraphs_nontokenized)
        print('Total Number of paragraphs: {}'.format(total_number_of_paragraphs))
        batch_counter = int(math.ceil(total_number_of_paragraphs / batch_size_for_p))
        print('Total Number of batches: {}'.format(batch_counter))
        print('Batch Size: {}'.format(batch_size_for_p))
        if is_generating_embeddings:
            execution_counter = 0
            while True:
                tf.reset_default_graph()
                elmo_embed = load_module("https://tfhub.dev/google/elmo/2", trainable=True)
                tf.logging.set_verbosity(tf.logging.ERROR)
                execution_counter += 1
                print(50 * '=')
                print('Exceution {}'.format(execution_counter))
                print(50 * '=')
                with tf.Session() as session:
                    paragraph_embeddings = None
                    timer_for_saving = save_for_every_batch
                    for i in range(1, batch_counter + 1):
                        if i * batch_size_for_p > total_number_of_paragraphs:
                            print('There is no more paragraphs left, batch counter is out of range {}/{}'.format(
                                i * batch_size_for_q, total_number_of_paragraphs))
                            if timer_for_saving != save_for_every_batch:
                                dump_embeddings(total_number_of_paragraphs,
                                                paragraph_embeddings_file + "_google_elmo_" + str(counter_for_saving))
                                print(paragraph_embeddings_file + "_google_elmo_" + str(counter_for_saving), ' is dumped')
                                del paragraph_embeddings
                                paragraph_embeddings = None
                                counter_for_saving += 1
                                batch_index_to_be_resumed += save_for_every_batch
                                timer_for_saving = save_for_every_batch
                                exit()
                        else:
                            if batch_index_to_be_resumed == -1 or (
                                    batch_index_to_be_resumed != -1 and i > batch_index_to_be_resumed):
                                # if there is still enough paragraphs for batch:
                                _begin_time = datetime.datetime.now()
                                print('Batch Index: {}'.format(i))
                                _begin_index = (i - 1) * batch_size_for_p
                                _end_index = _begin_index + batch_size_for_p
                                print('Paragraph Begin Index: {}'.format(_begin_index))
                                print('Paragraph End Index: {}'.format(_end_index))

                                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                                p_e = session.run(elmo_embed(
                                    inputs={
                                        "tokens": padded_tokenized_paragraph[_begin_index:_end_index],
                                        "sequence_len": paragraph_len[_begin_index:_end_index]
                                    },
                                    signature="tokens",
                                    as_dict=True)[embedding_type])

                                if paragraph_embeddings is None:
                                    paragraph_embeddings = np.mean(p_e, axis=1)
                                else:
                                    paragraph_embeddings = np.vstack((paragraph_embeddings, np.mean(p_e, axis=1)))

                                timer_for_saving -= 1
                                if timer_for_saving == 0:
                                    dump_embeddings(paragraph_embeddings,
                                                    paragraph_embeddings_file + "_google_elmo_" + str(counter_for_saving))
                                    print(paragraph_embeddings_file + "_google_elmo_" + str(counter_for_saving),
                                          ' is dumped')
                                    del paragraph_embeddings
                                    paragraph_embeddings = None
                                    counter_for_saving += 1
                                    batch_index_to_be_resumed += save_for_every_batch
                                    timer_for_saving = save_for_every_batch
                                    break
                                _end_time = datetime.datetime.now()
                                print('Loop {} takes {} mins'.format(i, (_end_time - _begin_time).seconds / 60))
                                print(50 * '+')
        else:
            all_paragraph_embeddings = None
            for i in range(1, counter_for_saving + 1):
                _all_paragraph_embeddings = load_embeddings(paragraph_embeddings_file + "_google_elmo_" + str(i))
                if all_paragraph_embeddings is None:
                    all_paragraph_embeddings = _all_paragraph_embeddings
                else:
                    all_paragraph_embeddings = np.vstack((all_paragraph_embeddings, _all_paragraph_embeddings))

            dump_embeddings(all_paragraph_embeddings, paragraph_embeddings_file + "_google_elmo")
            print(paragraph_embeddings_file + "_google_elmo", ' is dumped')
    end = datetime.datetime.now()
    print('Google ELMO Embeddings is completed in {} minutes'.format((end - start).seconds / 60))
    print(20 * '-')
