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
class generator:
    def __init__(self, file, additional_file, table_name='embeddings'):
        self.file = file
        self.additional_file = additional_file
        self.table_name = table_name
    def __call__(self):
        if self.additional_file is None:
            with h5py.File(self.file, 'r') as hf:
                for im in hf[self.table_name]:
                    yield im
        else:
            with h5py.File(self.file, 'r') as hf, h5py.File(self.additional_file, 'r') as af:
                af_ = np.reshape(af['embeddings'], [-1, 1])
                x = np.hstack((hf['embeddings'], af_))
                for im in x:
                    yield im

def get_dataset(question_embeddings_file,
                question_labels_file,
                paragraph_embeddings_file,
                embedding_dim,
                including_target=True):

    ques_ds = tf.data.Dataset.from_generator(
        generator(question_embeddings_file, additional_file=None),
        tf.float32,
        tf.TensorShape([embedding_dim,]))

    if including_target:
        parag_ds = tf.data.Dataset.from_generator(
            generator(paragraph_embeddings_file, additional_file=question_labels_file),
            tf.float32,
            tf.TensorShape([embedding_dim if question_labels_file is None else embedding_dim+1, ]))

        final_result = tf.data.Dataset.zip((ques_ds,parag_ds))
    else:
        final_result = ques_ds
    return final_result
