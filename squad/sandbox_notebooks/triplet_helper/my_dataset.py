#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the dataset."""

import h5py
import tensorflow as tf
import numpy as np
import random

class generator:
    def __init__(self, file, table_name='embeddings'):
        self.file = file
        self.table_name = table_name
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf[self.table_name]:
                yield im

def create_file_reader_ops(filename):
    reader = tf.data.TextLineDataset(filename).skip(1)
    record_defaults = [[0], [0]]
    COLUMNS = ['k', 'v']
    def _parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, record_defaults)

        # Pack the result into a dictionary
        features = dict(zip(COLUMNS, fields))

        # Separate the label from the features
        label = features.pop('v')

        return label

    return reader.map(_parse_line)

def get_dataset_from_cache(embeddings):
    return tf.data.Dataset.from_tensor_slices(embeddings)

def get_dataset(question_embeddings_file, paragraph_embeddings_file, label_file, q_embedding_shape, p_embedding_shape, including_label=True):

    ques_ds = tf.data.Dataset.from_generator(
        generator(question_embeddings_file),
        tf.float32,
        tf.TensorShape(q_embedding_shape))

    parag_ds = tf.data.Dataset.from_generator(
        generator(paragraph_embeddings_file),
        tf.float32,
        tf.TensorShape(p_embedding_shape))

    #ques_ds = ques_ds.concatenate(parag_ds)
    def decode_label(label):
        #label = tf.decode_raw(label, tf.uint16)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    if including_label:
        ps = create_file_reader_ops(label_file)
        #ps = ps.map(decode_label)
        final_result = tf.data.Dataset.zip((ques_ds,parag_ds))
    else:
        final_result = ques_ds
    return final_result


def _load_embeddings(infile_to_get):
    with h5py.File(infile_to_get, 'r') as fin:
        document_embeddings = fin['embeddings'][...]
    return document_embeddings

def get_question_embeddings(is_cached, question_embeddings, params):

    if not is_cached:
        _q = _load_embeddings(question_embeddings)
    else:
        _q = question_embeddings
    random.seed(params.eval_seed)
    qidx = random.sample(range(_q.shape[0]), params.eval_question_size_for_recall)
    _q = _q[qidx]
    #questions = tf.constant(_q)
    return _q

def get_embeddings(paragraph_embeddings_file):

    return tf.constant(_load_embeddings(paragraph_embeddings_file))