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
from glove import Glove, Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from shutil import copyfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bilm.elmo import ElmoEmbedder
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


def create_glove_corpus_model(corpus, model_path, window = 10, case_sensitive=True):

    corpus_model = Corpus()
    if case_sensitive:
        my_corpus = [[x.lower() for x in _ ]for _ in corpus]
    corpus_model.fit(my_corpus, window=window)
    corpus_model.save(model_path)
    return corpus_model

def create_glove_vector_model(corpus_model, model_path, dims = 100, learning_rate=0.05, epoch = 10,  threads = 10):

    glove_model = Glove(no_components=dims, learning_rate=learning_rate)
    glove_model.fit(corpus_model.matrix, epochs=epoch,
              no_threads=threads, verbose=True)
    glove_model.add_dictionary(corpus_model.dictionary)
    glove_model.save(model_path)
    return glove_model

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

def filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, predictions, paragraphs, data_eval, q_to_p, number_of_questions, outfile):
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
                        token_embeddings_file, voc_file_name, partition=20, weight_file = None, options_file = None):
    document_embedding_guideline = defaultdict()
    if not os.path.exists(token_embeddings_guideline_file):
        #########################
        ## use word embedding ##
        # ee = ElmoEmbedder(embedding_file=word_embeddings_file)
        ##########################
        ## use char encoding embedding ##
        ee = ElmoEmbedder(options_file=options_file, weight_file=weight_file)
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

        if partition > 1:
            is_first_record = True
            for partition_index in range(1, partition + 1):
                with h5py.File(token_embeddings_file.replace('@@', str(partition_index)), 'r') as fin:
                    print(partition_index)
                    token_embedding = fin['embeddings'][...]
                    if is_first_record:
                        document_embeddings = token_embedding
                        is_first_record = False
                    else:
                        document_embeddings = np.vstack((document_embeddings, token_embedding))
        else:
            with h5py.File(token_embeddings_file.replace('@@', str(1)), 'r') as fin:
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


TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = TRAIN
is_dump_during_execution = True
is_inject_idf = True
is_filtered_by_answers_from_rnet = True
is_split_content_to_documents = False
split_num_of_paragrahs_in_slices = 1000
percent_of_slice_splits = .4

# ELMO EMBEDDINGS #
is_elmo_document_embeddings_already_generated = False
partition_size = 5
is_elmo_word_embeddings_already_generated = False

# GLOVE TRAINING #
is_inject_local_weights = False
is_force_to_train_local_corpus = False
glove_window = 10
glove_dims = 1024
glove_learning_rate = 0.05
glove_epoch = 300
glove_threads = 10
local_embedding_models = ['glove']
################ CONFIGURATIONS #################


_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_paragraphs_file_name = '{}_paragraphs.txt'
paragraphs_file = os.path.join(datadir, _paragraphs_file_name)

_paragraph_embeddings_file_name = '{}_paragraph_embeddings.hdf5'.format(dataset_type)
paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

_token_embeddings_file_name = '{}_token_embeddings_@@.hdf5'.format(dataset_type)
token_embeddings_file = os.path.join(datadir, _token_embeddings_file_name)

_token_embeddings_guideline_file_name = '{}_token_embeddings_guideline.pkl'.format(dataset_type)
token_embeddings_guideline_file = os.path.join(datadir, _token_embeddings_guideline_file_name)

_questions_file_name = '{}_questions.txt'
questions_file = os.path.join(datadir, _questions_file_name)

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
                                                                           options_file=elmo_options_file
                                                                           )
        end = datetime.datetime.now()
        print('ELMO Token Embeddings is ended in {} minutes'.format((end-start).seconds/60))





        print('\n')
        print(20* '-')
        if is_inject_idf:
            print('IDF is going to be calculated')
            start = datetime.datetime.now()
            token2idfweight, idf_vec = transform_to_idf_weigths(tokenized_questions, tokenized_paragraphs, tokenize, questions_nontokenized, paragraphs_nontokenized)
            weighted_token_embeddings = np.multiply(idf_vec, token_embeddings)
            end = datetime.datetime.now()
            print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
        else:
            print('IDF is skipped')
            weighted_token_embeddings = token_embeddings







    print('\n')
    print(20 * '-')
    print('ELMO Embeddings is started')
    start = datetime.datetime.now()
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

    # for _token_embed_pack in [(weighted_token_embeddings, 'with_idf'), (token_embeddings, 'only')]:


    _token_embed = weighted_token_embeddings if not is_elmo_word_embeddings_already_generated else None
    _type = 'with_idf'
    for _a_b_c in _a_b_c_s:
        print('Weight {}'.format(_a_b_c))

        if not is_elmo_word_embeddings_already_generated:
            questions_embeddings, paragraphs_embeddings = token_to_document_embeddings(tokenized_questions,
                                                                                       tokenized_paragraphs,
                                                                                       _token_embed,
                                                                                       document_embedding_guideline)
            # YES TUNE
            WM = np.array([_a_b_c[0], _a_b_c[1], _a_b_c[2]]).reshape((1, 3, 1))
            questions_embeddings = np.multiply(questions_embeddings, WM)
            paragraphs_embeddings = np.multiply(paragraphs_embeddings, WM)

            questions_embeddings = np.mean(questions_embeddings, axis=1)
            paragraphs_embeddings = np.mean(paragraphs_embeddings, axis=1)
        else:
            with h5py.File(question_embeddings_file, 'r') as fq_in, h5py.File(paragraph_embeddings_file, 'r') as fp_in:
                paragraphs_embeddings = fp_in['embeddings'][...]
                questions_embeddings = fq_in['embeddings'][...]

        print('Nearest Neighbors: Starting')
        sub_start = datetime.datetime.now()
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
        if is_filtered_by_answers_from_rnet:
            with open(answers_file) as prediction_file:
                answers = json.load(prediction_file)

            filter_prediction_and_calculate_similarity_and_dump(paragraphs_embeddings, questions_embeddings, answers,
                                                                paragraphs, eval,
                                                                q_to_ps, len(questions),
                                                                os.path.join(datadir,
                                                                             'elmo_{}_weights_a_{}_b_{}_c_{}_output_filtered_answers_neighbors_###.csv'.format(
                                                                                 _type,
                                                                                 _a_b_c[
                                                                                     0],
                                                                                 _a_b_c[
                                                                                     1],
                                                                                 _a_b_c[
                                                                                     2])))
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
# TRAINING LOCAL WEIGHTS
if is_inject_local_weights:
    for _lm in local_embedding_models:
        local_paragraph_embeddings = file_pattern_for_local_weights.format(_lm, 'paragraph')
        local_question_embeddings = file_pattern_for_local_weights.format(_lm, 'question')
        local_corpus_model_file = file_pattern_for_local_model.format(_lm,'corpus')
        local_vector_model_file = file_pattern_for_local_model.format(_lm,'vector')
        if os.path.exists(glove_local_question_embeddings_file):
            if is_force_to_train_local_corpus:
                local_corpus_model = create_glove_corpus_model(corpus=tokenized_questions + tokenized_paragraphs,
                                       model_path=local_corpus_model_file,
                                       window=glove_window,
                                       case_sensitive=True)

                local_vector_model = create_glove_vector_model(local_corpus_model,
                                                                   local_vector_model_file,
                                                                   dims=glove_dims,
                                                                   learning_rate=glove_learning_rate,
                                                                   epoch=glove_epoch,
                                                                   threads=glove_threads)
                if is_inject_idf:
                    token2idfweight, idf_vec = transform_to_idf_weigths(tokenized_questions,
                                                                       tokenized_paragraphs,
                                                                       tokenize,
                                                                       questions_nontokenized,
                                                                       paragraphs_nontokenized)
                    glove_question_embeddings, glove_paragraph_embeddings = transorm_to_glove_embeddings(local_vector_model,
                                                             token2idfweight,
                                                             glove_dims,
                                                             tokenized_questions,
                                                             tokenized_paragraphs,
                                                             is_dump_during_execution,
                                                             glove_local_question_embeddings_file,
                                                             glove_local_paragraph_embeddings_file)

            else:
                local_corpus_model = Corpus.load(local_corpus_model_file)
                local_vector_model = Glove.load(local_vector_model_file)
                glove_question_embeddings = load_embeddings(glove_local_question_embeddings_file)
                glove_paragraph_embeddings = load_embeddings(glove_local_paragraph_embeddings_file)

        else:
            local_corpus_model = create_glove_corpus_model(corpus=tokenized_questions + tokenized_paragraphs,
                                                           model_path=local_corpus_model_file,
                                                           window=glove_window,
                                                           case_sensitive=True)

            local_vector_model = create_glove_vector_model(local_corpus_model,
                                                           local_vector_model_file,
                                                           dims=glove_dims,
                                                           learning_rate=glove_learning_rate,
                                                           epoch=glove_epoch,
                                                           threads=glove_threads)
            if is_inject_idf:
                token2idfweight, idf_vec = transform_to_idf_weigths(tokenized_questions,
                                                                    tokenized_paragraphs,
                                                                    tokenize,
                                                                    questions_nontokenized,
                                                                    paragraphs_nontokenized)
                glove_question_embeddings, glove_paragraph_embeddings = transorm_to_glove_embeddings(local_vector_model,
                                                                                                     token2idfweight,
                                                                                                     glove_dims,
                                                                                                     tokenized_questions,
                                                                                                     tokenized_paragraphs,
                                                                                                     is_dump_during_execution,
                                                                                                     glove_local_question_embeddings_file,
                                                                                                     glove_local_paragraph_embeddings_file)
        if is_filtered_by_answers_from_rnet:
            with open(answers_file) as prediction_file:
                answers = json.load(prediction_file)

        for _alpha in np.linspace(0.1, 0.9, 5):
            print('Nearest Neighbors: Starting')
            sub_start = datetime.datetime.now()

            final_paragraph_embeddings = (_alpha * paragraphs_elmo_embeddings) + ((1 - _alpha) * glove_paragraph_embeddings)
            final_question_embeddings = (_alpha * questions_elmo_embeddings) + ((1 - _alpha) * glove_question_embeddings)

            filter_prediction_and_calculate_similarity_and_dump(final_paragraph_embeddings, final_question_embeddings,
                                                                answers,
                                                                paragraphs, eval,
                                                                q_to_ps, len(questions),
                                                                os.path.join(datadir,
                                                                             'improved_elmo_with_glove_summed_weights_alpha_{}_output_filtered_answers_neighbors_###.csv'.format(
                                                                                 _alpha)))

            # final_paragraph_embeddings = (_alpha * paragraphs_elmo_embeddings) * (
            #             (1 - _alpha) * glove_paragraph_embeddings)
            # final_question_embeddings = (_alpha * questions_elmo_embeddings) * (
            #             (1 - _alpha) * glove_question_embeddings)
            #
            # filter_prediction_and_calculate_similarity_and_dump(final_paragraph_embeddings, final_question_embeddings,
            #                                                     answers,
            #                                                     paragraphs, eval,
            #                                                     q_to_ps, len(questions),
            #                                                     os.path.join(datadir,
            #                                                                  'improved_elmo_with_glove_multiplied_weights_alpha_{}_output_filtered_answers_neighbors_###.csv'.format(
            #                                                                      _alpha)))

            sub_end = datetime.datetime.now()
            print('Nearest Neighbors: Completed is completed in {} minutes'.format((sub_end - sub_start).seconds / 60))


end = datetime.datetime.now()
print('ELMO Embeddings is completed in {} minutes'.format((end - start).seconds / 60))
print(20 * '-')

