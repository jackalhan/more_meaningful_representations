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
dataset_type = DEV

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

_squad_file_name = '{}-v1.1.json'
squad_file = os.path.join(datadir, _squad_file_name)

OLD_API_ELMO={"is_inject_idf":True,
      "root_path": "ELMO_CONTEXT_OLD_API_EMBEDDINGS",
      "contextualized_document_embeddings_with_token": '{}_contextualized_document_embeddings_with_token.hdf5'.format(dataset_type),
      "is_paragraphs_listed_after_questions":True,
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


"""
******************************************************************************************************************
******************************************************************************************************************
START: DOCUMENT-TOKEN GUIDELINE
******************************************************************************************************************
******************************************************************************************************************
"""

tokenized_questions, tokenized_paragraphs = UTIL.fixing_the_token_problem(tokenized_questions, tokenized_paragraphs)
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
START: LOAD EMBEDINGS
******************************************************************************************************************
******************************************************************************************************************
"""
root_folder_path = os.path.join(datadir, args["root_path"])
document_embeddings = UTIL.load_embeddings(os.path.join(root_folder_path, args['contextualized_document_embeddings_with_token']))
print('contextualized_document_embeddings_with_token is loaded')

"""
******************************************************************************************************************
******************************************************************************************************************
END: LOAD EMBEDINGS
******************************************************************************************************************
******************************************************************************************************************
"""

"""
******************************************************************************************************************
******************************************************************************************************************
START: IDF
******************************************************************************************************************
******************************************************************************************************************
"""
if args['is_inject_idf']:
    print('IDF is going to be calculated')
    nlp = spacy.blank("en")
    tokenize = lambda doc: [token.text for token in nlp(doc)]
    start = datetime.datetime.now()
    token2idfweight, idf_vec = UTIL.transform_to_idf_weigths(tokenized_questions,
                                                             tokenized_paragraphs,
                                                             tokenize,
                                                             questions_nontokenized,
                                                             paragraphs_nontokenized)
    weighted_token_embeddings = np.multiply(idf_vec, document_embeddings)
    end = datetime.datetime.now()
    print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
else:
    print('IDF is skipped')
    _type = 'only'
    weighted_token_embeddings = document_embeddings
"""
******************************************************************************************************************
******************************************************************************************************************
END: LOAD IDF
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
del document_embeddings

print('Weighted are getting to applied documents with the following weights: {}'.format(args['weights_arguments']))

WM = np.array(args['weights_arguments']).reshape((1, len(args['weights_arguments']), 1))
weighted_token_embeddings = np.multiply(weighted_token_embeddings, WM)
weighted_token_embeddings = weighted_token_embeddings[:,0,:]

print('Weights are calculated according to the best combination')

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
del weighted_token_embeddings

with open(os.path.join(root_folder_path, args['word_vector_file_path'].replace("##", 'with_idf' if args['is_inject_idf'] else '')), 'a') as fout:
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