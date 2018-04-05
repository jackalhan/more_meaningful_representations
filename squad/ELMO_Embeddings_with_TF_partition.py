import datetime
import os
import re
import string
from collections import defaultdict, Counter
import pickle

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
from shutil import copyfile
import math
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bilm.elmo import ElmoEmbedder

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

_token_embeddings_file_name = '{}_token_embeddings_@@.hdf5'.format(dataset_type)
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

answers_file = os.path.join(datadir, '{}_answer.json'.format(dataset_type))


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
#for each question, paragraph index is added to question to paragraph
def filter_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, paragraphs, data_eval, slice_type, q_to_p, outfile):
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
                neighbor_list.append((slice_type,
                                      _id,
                                      neighbor_id,
                                      _fo + 1,
                                      #_ + 1,
                                      sk_sim[neighbor_id],
                                      q_to_p[_id],
                                      np.where(neighbors == q_to_p[_id])[0][0] + 1,
                                      sk_sim[q_to_p[_id]]
                                      ))
                _fo +=1


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

def filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, predictions, paragraphs, data_eval, slice_type, q_to_p, number_of_questions, outfile):
    neighbor_list_within_paragraph = []
    neighbor_list_match_answers = []
    for _id, _q_embedding in enumerate(tqdm(questions_embeddings, total=len(questions_embeddings))):
        _q_embedding = np.array([_q_embedding])
        sk_sim = cosine_similarity(_q_embedding, paragraphs_embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        ques_answers = data_eval[str(_id+1)]['answers']
        ques_id = data_eval[str(_id+1)]['uuid']
        pred_answer = predictions[ques_id]
        #p_id = q_to_p[_id]
        #neighbor_id = neighbors[p_id]
        for _, neighbor_id in enumerate(neighbors):
            is_answered_correctly = False
            if any_match_for_paragraph_and_answer(paragraphs[neighbor_id], ques_answers):
                is_answered_correctly = True

            neighbor_list_within_paragraph.append((slice_type,
                                  _id,
                                  neighbor_id,
                                  (q_to_p[_id] == neighbor_id),
                                  is_answered_correctly,
                                  sk_sim[neighbor_id],
                                  _ + 1,
                                  ))
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

    columns = ['slice_type', 'question', 'paragraph', 'ground_truth', 'is_model_answered_correctly',
                'cosine_score', 'nearest_neighbor_order']

    df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list_within_paragraph, columns=columns)
    #df_neighbor_match_answers= pd.DataFrame(data=neighbor_list_match_answers, columns=columns)

    df_neighbor_within_paragraph = df_neighbor_within_paragraph[df_neighbor_within_paragraph['is_model_answered_correctly'] == True]
    # df_neighbor_match_answers = df_neighbor_match_answers[
    #     df_neighbor_match_answers['is_model_answered_correctly'] == True]

    df_neighbor_within_paragraph.to_csv(outfile.replace('###','paragraph_contains_answer'),index=False)
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

    df_neighbor_within_paragraph_recalls.to_csv(outfile.replace('###', 'paragraph_contains_answer_recalls'), index=False)
    # df_neighbor_match_answers_recalls.to_csv(outfile.replace('###', 'answer_matches_answer_recalls'), index=False)

def calculate_recall_at_n(ns, data, number_of_questions):
    recalls = []
    for i in ns:
        total_number = len(data[data['nearest_neighbor_order'] <= i])
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
                        token_embeddings_file, voc_file_name, partition=20, partition_index=1):
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

        document_embeddings = ee.list_to_lazy_embeddings_with_dump(corpus_as_tokens, token_embeddings_file,partition)
    else:
        with open(token_embeddings_guideline_file, 'rb') as handle:
            document_embedding_guideline = pickle.load(handle)

        with h5py.File(token_embeddings_file.replace('@@', str(partition_index)), 'r') as fin:
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
# paragraphs, questions, q_to_p = read_squad_data(squad_other_file)
# paragraphs_test, questions_test, q_to_p_test = read_squad_data(squad_file)
train_word_counter, train_char_counter, dev_word_counter, dev_char_counter = Counter(), Counter(), Counter(), Counter()
dev_examples, dev_eval, dev_questions, dev_paragraphs, dev_q_to_ps = process_file(squad_other_file, "dev", dev_word_counter, dev_char_counter)
train_examples, train_eval, train_questions, train_paragraphs, train_q_to_ps = process_file(squad_file, "train", train_word_counter, train_char_counter)

end = datetime.datetime.now()
print('# of Paragraphs in Dev : {}'.format(len(dev_paragraphs)))
print('# of Questions in Dev: {}'.format(len(dev_questions)))
print('# of Q_to_P Dev: {}'.format(len(dev_q_to_ps)))

print('#' * 20)

print('# of Paragraphs in Train : {}'.format(len(train_paragraphs)))
print('# of Questions in Train: {}'.format(len(train_questions)))
print('# of Q_to_P in Train: {}'.format(len(train_q_to_ps)))
print('Squad Data: Reading Dev Ended in {} minutes'.format((end-start).seconds/60))


print(20* '-')
print('Paragraphs: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_paragraphs = tokenize_contexts(dev_paragraphs)
tokenized_train_paragraphs = tokenize_contexts(train_paragraphs)
#dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file)
#dump_tokenized_contexts(tokenized_train_paragraphs, paragraphs_file)
end = datetime.datetime.now()
print('# of Tokenized Paragraphs: {}'.format(len(tokenized_paragraphs)))
print('Paragraphs: Tokenization and Saving Tokenization  is Completed in {} minutes'.format((end-start).seconds/60))

print(20* '-')
print('Questions: Tokenization and Saving Tokenization Started')
start = datetime.datetime.now()
tokenized_questions = tokenize_contexts(dev_questions)
tokenized_train_questions = tokenize_contexts(train_questions)
#dump_tokenized_contexts(tokenized_questions,questions_file)
#dump_tokenized_contexts(tokenized_train_questions, questions_file)
end = datetime.datetime.now()
print('# of Tokenized Questions: {}'.format(len(tokenized_questions)))
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
questions_train_nontokenized = [" ".join(context) for context in tokenized_train_questions]
paragraphs_train_nontokenized = [" ".join(context) for context in tokenized_train_paragraphs]


tfidf = TfidfVectorizer(analyzer=lambda x: x, smooth_idf=False, sublinear_tf=False, tokenizer=tokenize)
tfidf.fit(questions_nontokenized+paragraphs_nontokenized + questions_train_nontokenized + paragraphs_train_nontokenized)
max_idf = max(tfidf.idf_)
token2idfweight = defaultdict(
    lambda: max_idf,
    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

# tfidf_paragraphs = np.array(tfidf.transform(tokenized_paragraphs).toarray().tolist())
# tfidf_questions = np.array(tfidf.transform(tokenized_questions).toarray().tolist())

print(20* '-')
print('ELMO Token Embeddings is started')

# -----------------------------
# SAMPLING RATE FOR BIGGER DATASETS
# -----------------------------
# sampling_rate = 0.05
# _qs_len = len(tokenized_train_questions)
# _ps_len = len(tokenized_train_paragraphs)
# qs_len_to_retrieve = int(_qs_len * sampling_rate)
# ps_len_to_retrieve = int(_ps_len * sampling_rate)
# tokenized_train_questions= tokenized_train_questions[0:qs_len_to_retrieve]
# tokenized_train_paragraphs= tokenized_train_paragraphs[0:ps_len_to_retrieve]
# print(20* '-')
# print('# of New Train Sampling Tokenized Paragraphs: {}'.format(len(tokenized_train_paragraphs)))
# print('# of New Train Sampling Tokenized Questions: {}'.format(len(tokenized_train_questions)))
# # -----------------------------
# -----------------------------

start = datetime.datetime.now()

with open(token_embeddings_guideline_file, 'rb') as handle:
    document_embedding_guideline = pickle.load(handle)

total_tokens_in_each_embedding_file = 715372

par_str_doc_first_index = len(tokenized_train_questions)
par_str_doc_last_index = len(tokenized_train_questions + tokenized_train_paragraphs) - 1

par_token_str_index = document_embedding_guideline[par_str_doc_first_index]['start_index']
par_token_end_index = document_embedding_guideline[par_str_doc_last_index]['end_index']

partitioned_embs_files_start_indx = math.ceil(par_token_str_index/total_tokens_in_each_embedding_file)
partitioned_embs_files_end_indx = math.ceil(par_token_end_index/total_tokens_in_each_embedding_file)

is_first_record = True
#for partition_index in range(partitioned_embs_files_start_indx, partitioned_embs_files_end_indx + 1):
for partition_index in range(1, 5 + 1):
    with h5py.File(token_embeddings_file.replace('@@', str(partition_index)), 'r') as fin:
        print(partition_index)
        token_embedding = fin['embeddings'][...]
        if is_first_record:
            token_embeddings = token_embedding
            is_first_record = False
        else:
            token_embeddings = np.vstack((token_embeddings, token_embedding))


print('Token Embeddings Done')

end = datetime.datetime.now()
print('ELMO Token Embeddings is ended in {} minutes'.format((end-start).seconds/60))
# # WEIGHT MATRIX FOR TUNING
# # a = .3
# # b = 1-a
# # WM = np.array([1, a, b]).reshape((1,3,1))
#
# IDF MATRIX SHAPE OF [x, 1, k, 1], where x = number of documents, k = max length of document
# IDFM =


idf_vec = create_idf_matrix(tokenized_train_questions, tokenized_train_paragraphs, token2idfweight)
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

#for _token_embed_pack in [(idf_weighted_token_embeddings, 'with_idf'), (token_embeddings, 'only')]:
for _token_embed_pack in [(idf_weighted_token_embeddings, 'with_idf')]:
    _token_embed = _token_embed_pack[0]
    _type = _token_embed_pack[1]

    start = datetime.datetime.now()
    print('ELMO Embeddings is started for "{}" type'.format(_type))

    for _a_b_c in _a_b_c_s:
        print('Weight {}'.format(_a_b_c))
        questions_embeddings, paragraphs_embeddings = token_to_document_embeddings(tokenized_train_questions,
                                                                                   tokenized_train_paragraphs, _token_embed,
                                                                                   document_embedding_guideline)
        # YES TUNE
        WM = np.array([_a_b_c[0], _a_b_c[1], _a_b_c[2]]).reshape((1, 3, 1))
        questions_embeddings = np.multiply(questions_embeddings, WM)
        paragraphs_embeddings = np.multiply(paragraphs_embeddings, WM)

        questions_embeddings = np.mean(questions_embeddings, axis=1)
        paragraphs_embeddings = np.mean(paragraphs_embeddings, axis=1)
        print('Nearest Neighbors: Starting')
        # calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, s['slice_type'], dev_q_to_ps,
        #                               os.path.join(datadir, 'elmo_{}_weights_a_{}_b_{}_c_{}_output_neighbors.csv'.format(_type, _a_b_c[0], _a_b_c[1], _a_b_c[2])))

        # filter_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, dev_paragraphs, dev_eval, s['slice_type'],dev_q_to_ps,
        #                               os.path.join(datadir,
        #                                            'elmo_{}_weights_a_{}_b_{}_c_{}_output_filtered_neighbors.csv'.format(_type,
        #                                                                                                         _a_b_c[
        #                                                                                                             0],
        #                                                                                                         _a_b_c[
        #                                                                                                             1],
        #                                                                                                         _a_b_c[
        #                                                                                                             2])))

        with open(answers_file) as prediction_file:
            answers = json.load(prediction_file)

        filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, answers, train_paragraphs, train_eval,
                                                 s['slice_type'], train_q_to_ps, len(train_questions),
                                                 os.path.join(datadir,
                                                              'elmo_{}_weights_a_{}_b_{}_c_{}_output_filtered_answers_neighbors_###.csv'.format(
                                                                  _type,
                                                                  _a_b_c[
                                                                      0],
                                                                  _a_b_c[
                                                                      1],
                                                                  _a_b_c[
                                                                      2])))
        print('Nearest Neighbors: Completed')
    end = datetime.datetime.now()
    print('ELMO Embeddings is completed in {} minutes for "{}" type'.format((end - start).seconds / 60, _type))
    print(20 * '-')
questions_embeddings = np.reshape(questions_embeddings, (questions_embeddings.shape[0], questions_embeddings.shape[1]))
dump_embeddings(questions_embeddings, question_embeddings_file)
paragraphs_embeddings = np.reshape(paragraphs_embeddings, (paragraphs_embeddings.shape[0], paragraphs_embeddings.shape[1]))
dump_embeddings(paragraphs_embeddings, paragraph_embeddings_file)

print('All Done!!!!!!')


# print('Nearest Neighbors: Starting')
# start = datetime.datetime.now()
# calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, s['slice_type'], q_to_p, os.path.join(datadir, 'elmo_with_idf_output_neighbors.csv'))
# #calculate_similarity_and_dump(idf_injected_paragraph_embeddings, idf_injected_question_embeddings, s['slice_type'], q_to_p, os.path.join(datadir, 'elmo_with_idf_output_neighbors.csv'))
# end = datetime.datetime.now()
# print('Nearest Neighbors: Completed')


