import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
import argparse
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--partition_size', type=int, default=10000)
    return parser


def load_data(embedding_path, label_path, prefix):
    question_embeddings = UTIL.load_embeddings(os.path.join(embedding_path, prefix + '_question_embeddings.hdf5'))
    paragraph_embeddings = UTIL.load_embeddings(os.path.join(embedding_path, prefix + '_paragraph_embeddings.hdf5'))
    labels = pd.read_csv(os.path.join(label_path, prefix + '_question_labels.csv'))
    return question_embeddings, paragraph_embeddings, labels
def dump_splitted_train_test(question_embeddings, paragraph_embeddings, labels, prefix, path, partition_size):
    UTIL.dump_embeddings(labels['q'],
                    os.path.join(path, prefix + "_question_idx.hdf5"))
    UTIL.dump_embeddings(labels['p'],
                         os.path.join(path, prefix + "_question_labels.hdf5"),dtype='int32')
    range_size = math.ceil(question_embeddings.shape[0]/partition_size)
    for part in range(0, range_size):
        pair_paragraph_embeddings = None
        start = part * partition_size
        end = start + partition_size
        for q_indx, q_embed in tqdm(enumerate(question_embeddings[start:end])):
            if pair_paragraph_embeddings is None:
                pair_paragraph_embeddings = paragraph_embeddings[labels['p'][q_indx]]
            else:
                pair_paragraph_embeddings = np.vstack(
                    (pair_paragraph_embeddings, paragraph_embeddings[labels['p'][q_indx]]))
        UTIL.dump_embeddings(pair_paragraph_embeddings, os.path.join(path, prefix + "_paired_paragraph_embeddings_part_{}.hdf5".format(part)))

    pair_paragraph_embeddings = None
    for part in range(0, range_size):
        embeddings = UTIL.load_embeddings(os.path.join(path, prefix + "_paired_paragraph_embeddings_part_{}.hdf5".format(part)))
        if pair_paragraph_embeddings is None:
            pair_paragraph_embeddings = embeddings
        else:
            pair_paragraph_embeddings = np.vstack(
                (pair_paragraph_embeddings,embeddings))

    UTIL.dump_embeddings(pair_paragraph_embeddings,
                         os.path.join(path, prefix + "_paired_paragraph_embeddings.hdf5"))

    for part in range(0, range_size):
        os.remove(os.path.join(path, prefix + "_paired_paragraph_embeddings_part_{}.hdf5".format(part)))

    UTIL.dump_embeddings(question_embeddings,
                    os.path.join(path,prefix + '_question_embeddings.hdf5'))
    UTIL.dump_embeddings(paragraph_embeddings, os.path.join(path, prefix + "_all_paragraph_embeddings.hdf5"))
def main(args):
    path = UTIL.create_dir(os.path.join(args.embedding_path, 'splitted_train_test'))
    test_question_embeddings, test_paragraph_embeddings, test_labels = load_data(args.embedding_path, args.label_path, 'test')
    dump_splitted_train_test(test_question_embeddings, test_paragraph_embeddings, test_labels, 'test', path, args.partition_size)
    print('Test data is ready')
    train_question_embeddings, train_paragraph_embeddings, train_labels = load_data(args.embedding_path, args.label_path,
                                                                              'train')
    dump_splitted_train_test(train_question_embeddings, train_paragraph_embeddings, train_labels, 'train', path, args.partition_size)
    print('Train data is ready')

if __name__ == '__main__':
    """
    sample executions: 

    """
    args = get_parser().parse_args()
    main(args)
