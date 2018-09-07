import datetime
from collections import Counter, defaultdict
import tensorflow as tf
#from tqdm import tqdm
#from glove import Glove, Corpus
import chakin
import numpy as np
import os
import sys
import spacy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
TRAIN = 'train'
DEV = 'dev'

################ CONFIGURATIONS #################
dataset_type = DEV
is_dump_during_execution = False
is_inject_idf = False
is_tf_style = False
################ CONFIGURATIONS #################

_basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
datadir = os.path.join(_basepath, dataset_type)

pre_trained_dir = UTIL.create_dir(os.path.join(_basepath, 'GLOVE', 'data'))
paragraphs_dir = UTIL.create_dir(os.path.join(datadir, 'GLOVE', 'paragraphs'))
questions_dir = UTIL.create_dir(os.path.join(datadir, 'GLOVE','questions'))

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

_voc_file_name = '{}_voc.txt'.format(dataset_type)
voc_file_name = os.path.join(datadir, _voc_file_name)


# question_embedddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/test_question_5000_embeddings.hdf5'
# paragraph_embeddings_path = '/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/data/recall_test/recall_paragraph_embeddings.hdf5'
# question_indx_path = ''

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
tokenized_questions, tokenized_paragraphs = UTIL.fixing_the_token_problem(tokenized_questions, tokenized_paragraphs)
document_embedding_guideline, corpus_as_tokens = UTIL.generate_document_embedding_guideline(tokenized_questions, tokenized_paragraphs, is_dump_during_execution, token_embeddings_guideline_file, tokens_ordered_file)
all_tokens = set(['<S>', '</S>','<UNK>'])
for token in corpus_as_tokens:
    all_tokens.add(token)
with open(voc_file_name, 'w') as fout:
    fout.write('\n'.join(all_tokens))

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
START: GLOVE EMBEDDINGS DOWNLOAD
******************************************************************************************************************
******************************************************************************************************************
"""

print(100 * '*')
print('Downloading Glove Pre-Trained Embeddings....')
start = datetime.datetime.now()

CHAKIN_INDEX = 16
NUMBER_OF_DIMENSIONS = 300
SUBFOLDER_NAME = "GloVe.840B.300d"

ZIP_FILE = os.path.join(pre_trained_dir, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(pre_trained_dir, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))

if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    # GloVe by Stanford is licensed Apache 2.0:
    #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
    #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
    #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
    print("Downloading embeddings to '{}'".format(ZIP_FILE))
    chakin.download(number=CHAKIN_INDEX, save_dir='./{}'.format(pre_trained_dir))
else:
    print("Embeddings already downloaded.")

if not os.path.exists(UNZIP_FOLDER):
    import zipfile

    if not os.path.exists(ZIP_FILE) and os.path.exists(ZIP_FILE_ALT):
        ZIP_FILE = ZIP_FILE_ALT
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
        zip_ref.extractall(UNZIP_FOLDER)
else:
    print("Embeddings already extracted.")
end = datetime.datetime.now()
print('ELMO Embeddings from Glove are generated in {} minutes'.format((end - start).seconds / 60))
print(100 * '*')
"""
******************************************************************************************************************
******************************************************************************************************************
END: GLOVE EMBEDDINGS DOWNLOAD
******************************************************************************************************************
******************************************************************************************************************
"""

"""
******************************************************************************************************************
******************************************************************************************************************
START: GLOVE EMBEDDINGS LOAD
******************************************************************************************************************
******************************************************************************************************************
"""
def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    with open(glove_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict

"""
******************************************************************************************************************
******************************************************************************************************************
END: GLOVE EMBEDDINGS LOAD
******************************************************************************************************************
******************************************************************************************************************
"""


"""
******************************************************************************************************************
******************************************************************************************************************
START : GLOVE CLASSICAL STYLE
******************************************************************************************************************
******************************************************************************************************************
"""
if not is_tf_style:
    def read_file(file_name):
        with open(file_name) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content


    encoding = "utf-8"
    dim = 300
    tokens = []
    if voc_file_name is not None:
        tokens = read_file(voc_file_name)
    glove_word_weights = {}
    with open(GLOVE_FILENAME, "rb") as infile:
        for line in infile:
            parts = line.split()
            token = parts[0].decode(encoding)
            if len(tokens) > 0:
                if token in tokens:
                    nums = np.array(parts[1:], dtype=np.float32)
                    glove_word_weights[token] = nums
            else:
                nums = np.array(parts[1:], dtype=np.float32)
                glove_word_weights[token] = nums

    if is_inject_idf:
        print('IDF is going to be calculated')
        nlp = spacy.blank("en")
        tokenize = lambda doc: [token.text for token in nlp(doc)]
        start = datetime.datetime.now()
        token2idfweight, idf_vec = UTIL.transform_to_idf_weigths(tokenized_questions,
                                                                 tokenized_paragraphs,
                                                                 tokenize,
                                                                 questions_nontokenized,
                                                                 paragraphs_nontokenized)

        mean_glove_with_idf_embeddings = np.array([
            np.mean([glove_word_weights[w] * token2idfweight[w]
                     for w in words if w in glove_word_weights] or
                    [np.zeros(dim)], axis=0)
            for words in tokenized_questions + tokenized_paragraphs
        ])
        # UTIL.dump_embeddings(mean_glove_with_idf_embeddings,
        #                 os.path.join(datadir, 'dev_mean_glove_with_idf_embeddings.hdf5'))
        question_embeddings = mean_glove_with_idf_embeddings[0:len(tokenized_questions), :]
        UTIL.dump_embeddings(question_embeddings, os.path.join(datadir, '{}_glove_questions_embeddings_with_idf.hdf5'.format(dataset_type)))
        paragraphs_embeddings = mean_glove_with_idf_embeddings[len(tokenized_questions):, :]
        UTIL.dump_embeddings(paragraphs_embeddings, os.path.join(datadir, '{}_glove_paragraphs_embeddings_with_idf.hdf5'.format(dataset_type)))
        end = datetime.datetime.now()
        print('IDF calculation is ended in {} minutes'.format((end - start).seconds / 60))
    else:
        print('IDF is skipped')
        mean_glove_embeddings = np.array([
            np.mean([glove_word_weights[w] for w in words if w in glove_word_weights]
                    or [np.zeros(dim)], axis=0)
            for words in tokenized_questions + tokenized_paragraphs
        ])
        #UTIL.dump_embeddings(mean_glove_embeddings, os.path.join(datadir, 'dev_mean_glove_embeddings.hdf5'))
        question_embeddings = mean_glove_embeddings[0:len(tokenized_questions), :]
        UTIL.dump_embeddings(question_embeddings, os.path.join(datadir, '{}_glove_questions_embeddings.hdf5'.format(dataset_type)))
        paragraphs_embeddings = mean_glove_embeddings[len(tokenized_questions):, :]
        UTIL.dump_embeddings(paragraphs_embeddings,
                             os.path.join(datadir, '{}_glove_paragraphs_embeddings.hdf5'.format(dataset_type)))


"""
******************************************************************************************************************
******************************************************************************************************************
END : GLOVE CLASSICAL STYLE
******************************************************************************************************************
******************************************************************************************************************
"""

"""
******************************************************************************************************************
******************************************************************************************************************
START: GLOVE EMBEDDINGS TO DOCUMENT EMBEDDINGS WITH TF STYLE
******************************************************************************************************************
*******
"""

if is_tf_style:
    print("Loading embedding from disks...")
    word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
    print("Embedding loaded from disks.")

    vocab_size, embedding_dim = index_to_embedding.shape
    print("Embedding is of shape: {}".format(index_to_embedding.shape))
    print("This means (number of words, number of dimensions per word)\n")
    print("The first words are words that tend occur more often.")

    print("Note: for unknown words, the representation is an empty vector,\n"
          "and the index is the last one. The dictionnary has a limit:")
    print("    {} --> {} --> {}".format("A word", "Index in embedding", "Representation"))
    word = "worsdfkljsdf"
    idx = word_to_index[word]
    embd = list(np.array(index_to_embedding[idx], dtype=int))  # "int" for compact print only.
    print("    {} --> {} --> {}".format(word, idx, embd))
    word = "the"
    idx = word_to_index[word]
    embd = list(index_to_embedding[idx])  # "int" for compact print only.
    print("    {} --> {} --> {}".format(word, idx, embd))

    batch_size = None  # Any size is accepted

    tf.reset_default_graph()

    # Define the variable that will hold the embedding:
    tf_embedding = tf.Variable(
        tf.constant(0.0, shape=index_to_embedding.shape),
        trainable=False,
        name="Embedding"
    )

    tf_word_ids = tf.placeholder(tf.int32, shape=[batch_size])

    tf_word_representation_layer = tf.nn.embedding_lookup(
        params=tf_embedding,
        ids=tf_word_ids
    )

    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=index_to_embedding.shape)
    tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

    batch_of_words = tokenized_questions[0]
    batch_indexes = [word_to_index[w] for w in batch_of_words]

    with tf.Session() as sess:#sess = tf.InteractiveSession()
        _ = sess.run(
            tf_embedding_init,
            feed_dict={
                tf_embedding_placeholder: index_to_embedding
            }
        )
        print("Embedding now stored in TensorFlow. Can delete numpy array to clear some RAM.")
        del index_to_embedding


        embedding_from_batch_lookup = sess.run(
            tf_word_representation_layer,
            feed_dict={
                tf_word_ids: batch_indexes
            }
        )
        print("Representations for {}:".format(batch_of_words))
        print(embedding_from_batch_lookup)
        print(embedding_from_batch_lookup.shape)

"""
******************************************************************************************************************
******************************************************************************************************************
END: GLOVE EMBEDDINGS TO DOCUMENT EMBEDDINGS WITH TF STYLE
******************************************************************************************************************
*******
"""