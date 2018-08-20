import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import h5py

def load_module(module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"):
    embed = hub.Module(module_url)
    return embed

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

if __name__ == '__main__':

    # STARTING PARAMETERS
    TRAIN = 'train'
    DEV = 'dev'
    dataset_type = DEV
    _basepath = os.path.abspath(__file__).rpartition(os.sep)[0]
    datadir = os.path.join(_basepath, dataset_type)

    _questions_file_name = '{}_questions.txt'
    questions_file = os.path.join(datadir, _questions_file_name)
    _question_embeddings_file_name = '{}_USE_question_embeddings.hdf5'.format(dataset_type)
    question_embeddings_file = os.path.join(datadir, _question_embeddings_file_name)

    _paragraphs_file_name = '{}_paragraphs.txt'
    paragraphs_file = os.path.join(datadir, _paragraphs_file_name)
    _paragraph_embeddings_file_name = '{}_USE_paragraph_embeddings.hdf5'.format(dataset_type)
    paragraph_embeddings_file = os.path.join(datadir, _paragraph_embeddings_file_name)

    # ENDING PARAMETERS

    # INITIALIZE USE
    embed = load_module()
    questions = read_file(questions_file.format(dataset_type))
    paragraphs = read_file(paragraphs_file.format(dataset_type))
    print("Question Len", len(questions))
    print("Paragraphs Len", len(paragraphs))
    #word = "Elephant"
    sentence = "the more 'diluted' the embedding will be."
    sentence2 = "the more ' diluted ' the embedding will be ."
    # paragraph = (
    #     "Universal Sentence Encoder embeddings also support short paragraphs. "
    #     "There is no hard limit on how long the paragraph is. Roughly, the longer "
    #     "the more 'diluted' the embedding will be.")
    messages = [sentence, sentence2]

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(embed(messages))

      x = None
      for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
          print("Message: {}".format(messages[i]))
          print("Embedding size: {}".format(len(message_embedding)))
          if x == None:
              x = message_embedding
          else:
              if x == message_embedding:
                  print('Identical')
          message_embedding_snippet = ", ".join(
              (str(x) for x in message_embedding[:3]))
          print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

