import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import spacy
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = DEV

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

question_embedddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/test_question_5000_embeddings.hdf5'
paragraph_embeddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/recall_paragraph_embeddings.hdf5'
question_labels_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/test_question_5000_labels.hdf5'
recalls_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/recalls_###.csv'
is_from_scratch = False ## Whether you want to fill the p_to_q list from a squad txt file or manually provided by question_labels_path


def calculate_recall_at_n(ns, data, number_of_questions):
    recalls = []
    for i in ns:
        total_number = len(data[(data['nearest_neighbor_order'] < i) & (data['ground_truth'] == True) ])
        recalls.append((i, total_number, total_number/number_of_questions))
    return recalls

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


if is_from_scratch:

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
    #
    # if is_dump_during_execution:
    #     UTIL.dump_tokenized_contexts(tokenized_paragraphs, paragraphs_file.format(dataset_type))
    #     UTIL.dump_tokenized_contexts(tokenized_questions, questions_file.format(dataset_type))
    #     UTIL.dump_mapping_data(q_to_ps, mapping_file.format(dataset_type))
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

    tokenized_questions, tokenized_paragraphs = UTIL.fixing_the_token_problem(tokenized_questions, tokenized_paragraphs)

else:

    q_to_ps = UTIL.load_embeddings(question_labels_path)

paragraphs_embeddings = UTIL.load_embeddings(paragraph_embeddings_path)
questions_embeddings = UTIL.load_embeddings(question_embedddings_path)
calculate_similarity_and_dump(paragraphs_embeddings,
                                      questions_embeddings,
                                      q_to_ps,
                                      len(questions_embeddings),
                                      recalls_path)