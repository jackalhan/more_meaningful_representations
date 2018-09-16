import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import math
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
partition = 50
left_off = 17
is_all_done=True
source_embedddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/improved_question_embeddings.hdf5'
target_embeddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/paragraph_embeddings.hdf5'
labels_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/question_labels.hdfs'
recalls_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/recalls_###.csv'
is_from_scratch = False ## Whether you want to fill the p_to_q list from a squad txt file or manually provided by question_labels_path


def calculate_recall_at_n(ns, data, number_of_sources):
    recalls = []
    for i in ns:
        total_number = len(data[(data['nearest_neighbor_order'] < i) & (data['ground_truth'] == True) ])
        recalls.append((i, total_number, total_number/number_of_sources))
    return recalls

def calculate_similarity_and_dump(target_embeddings,
                                  source_embeddings,
                                  s_to_t,
                                  number_of_sources,
                                  outfile):
    columns = ['source', 'target', 'ground_truth', 'is_model_answered_correctly',
               'cosine_score', 'nearest_neighbor_order']
    neighbor_list = []
    partition_size = math.ceil(source_embeddings.shape[0] / partition)
    partition_counter = 1
    print('Each partition has {} size for total {} records'.format(partition_size, source_embeddings.shape[0]))
    if not is_all_done:
        for _id, _q_embedding in enumerate(tqdm(source_embeddings, total=len(source_embeddings))):
            if partition_counter < left_off:
                partition_counter += 1
                continue
            _q_embedding = np.array([_q_embedding])
            sk_sim = cosine_similarity(_q_embedding, target_embeddings)[0]
            neighbors = np.argsort(-sk_sim)
            for _, neighbor_id in enumerate(neighbors):
                neighbor_list.append((_id,
                                      neighbor_id,
                                      (s_to_t[_id] == neighbor_id),
                                      True,
                                      sk_sim[neighbor_id],
                                      _,
                                      ))
            if _id == partition_size*partition_counter:
                print('Partition {} is completed'.format(partition_counter))
                df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list, columns=columns)
                df_neighbor_within_paragraph.to_csv(outfile.replace('###', 'partition_' + str(partition_counter)), index=False)
                partition_counter +=1
                neighbor_list = []

        if neighbor_list:
            print('Last Partition {} is also completed'.format(partition_counter))
            df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_list, columns=columns)
            df_neighbor_within_paragraph.to_csv(outfile.replace('###', 'partition_' + str(partition_counter)), index=False)
            partition_counter += 1
            neighbor_list = []

    neighbor_within_paragraph = []
    for part_counter in range(1, partition+1):
        neighbor_within_paragraph.append(pd.read_csv(outfile.replace('###', 'partition_' + str(part_counter))))
        print("{} is completed".format(part_counter))

    df_neighbor_within_paragraph = pd.DataFrame(data=neighbor_within_paragraph, columns=columns)
    df_neighbor_within_paragraph = df_neighbor_within_paragraph[
        df_neighbor_within_paragraph['is_model_answered_correctly'] == True]

    df_neighbor_within_paragraph.to_csv(outfile.replace('###', ''), index=False)
    recall_ns = [1, 2, 5, 10, 20, 50]
    recall_columns = ['n', 'number_of_true', 'normalized_recalls']
    df_neighbor_within_paragraph_recalls = pd.DataFrame(data=calculate_recall_at_n(recall_ns,
                                                                                   df_neighbor_within_paragraph,
                                                                                   number_of_sources)
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

    s_to_ts = UTIL.load_embeddings(labels_path).astype(int)

target_embeddings = UTIL.load_embeddings(target_embeddings_path)
source_embeddings = UTIL.load_embeddings(source_embedddings_path)
calculate_similarity_and_dump(target_embeddings,
                                      source_embeddings,
                                      s_to_ts,
                                      len(source_embeddings),
                                      recalls_path)