import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
import csv
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = TRAIN

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

OLD_API_ELMO={
      "is_inject_idf":False,
      "total_number_of_partitioned_files": 36,
      "root_path": "ELMO_CONTEXT_OLD_API_EMBEDDINGS",
       "calculated_token_embeddings_file": '{}_contextualized_document_embeddings_with_token_##_@@.hdf5'.format(dataset_type),
      "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token.hdf5'.format(dataset_type),
      "weights_arguments": [1, 0, 0],
      "word_vector_file_path" : '{}_word_embeddings_##.txt'.format(dataset_type)
      }

args = OLD_API_ELMO


################ CONFIGURATIONS #################


"""
******************************************************************************************************************
******************************************************************************************************************
START: PARSING FILE
******************************************************************************************************************
******************************************************************************************************************
"""

tokenized_questions, tokenized_paragraphs, questions_nontokenized, paragraphs_nontokenized = UTIL.prepare_squad_objects(squad_file.format(dataset_type),dataset_type)
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

document_embedding_guideline, corpus_as_tokens = UTIL.generate_document_embedding_guideline(tokenized_questions, tokenized_paragraphs)

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
START: WEIGHTED ARE GETTING APPLIED TO TOKEN EMBEDDINGS
******************************************************************************************************************
******************************************************************************************************************
"""
root_folder_path = os.path.join(datadir, args["root_path"])

weighted_token_embeddings = None
for partition in range(1, args['total_number_of_partitioned_files']+1):
    temp_weighted_token_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args[
        'calculated_token_embeddings_file'].replace('@@', str(partition)).replace('##', 'idf' if args['is_inject_idf'] else '')))
    # WM = np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
    # print('Weights are calculated according to the best combination')
    # print('temp_weighted_token_embeddings shape {}, WM shape {}'.format(temp_weighted_token_embeddings.shape, WM.shape))
    # temp_weighted_token_embeddings = np.multiply(temp_weighted_token_embeddings, WM)
    # print('temp_weighted_token_embeddings shape {}'.format(temp_weighted_token_embeddings.shape))
    # temp_weighted_token_embeddings = temp_weighted_token_embeddings[:, 0, :]
    # print('temp_weighted_token_embeddings shape {}'.format(temp_weighted_token_embeddings.shape))
    if weighted_token_embeddings is None:
        weighted_token_embeddings = temp_weighted_token_embeddings
    else:
        weighted_token_embeddings = np.vstack((weighted_token_embeddings, temp_weighted_token_embeddings))
    print("Partition {} is loaded".format(partition))


print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

"""
******************************************************************************************************************
******************************************************************************************************************
END: WEIGHTED ARE GETTING APPLIED TO TOKEN EMBEDDINGS
******************************************************************************************************************
******************************************************************************************************************
"""

"""
******************************************************************************************************************
******************************************************************************************************************
START: GENERATE WEIGHT FILE
******************************************************************************************************************
******************************************************************************************************************
"""
print('Weight file is generated')
start = datetime.datetime.now()
token2elmoweight=dict()
weighted_token_embeddings_and_token = zip(corpus_as_tokens, weighted_token_embeddings)
del weighted_token_embeddings, corpus_as_tokens

with open(os.path.join(root_folder_path, args['word_vector_file_path'].replace("##", 'with_idf' if args['is_inject_idf'] else '' )), 'a') as fout:
    #writer = csv.writer(fout, lineterminator='\n')
    for line in weighted_token_embeddings_and_token:
        token = line[0]
        data = line[1]
        if token not in token2elmoweight:
            #data = data.reshape(1, -1)
            data = data.tolist()
            data = ' '.join(map(str, data))
            token2elmoweight[token] = 1
            fout.write(token + ' ' + data)
            fout.write("\n")

end = datetime.datetime.now()
print('Done in {} minutes'.format((end - start).seconds / 60))
"""
******************************************************************************************************************
******************************************************************************************************************
END: GENERATE WEIGHT FILE
******************************************************************************************************************
******************************************************************************************************************
"""