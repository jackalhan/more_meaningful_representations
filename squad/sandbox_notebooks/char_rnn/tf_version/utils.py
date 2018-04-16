import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, data_file, batch_size, seq_length, is_forced = False, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding=encoding
        self.data_file_pre = data_file.split('.')[0]
        vocab_file_name = '{}_vocab.pkl'.format(self.data_file_pre)
        tensor_file_name = '{}_data.pkl'.format(self.data_file_pre)
        self.input_file = os.path.join(self.data_dir, data_file)
        self.vocab_file= os.path.join(self.data_dir, vocab_file_name)
        self.tensor_file = os.path.join(self.data_dir, tensor_file_name)

        if not os.path.exists(self.vocab_file) and not os.path.exists(self.tensor_file):
            print('Reading a file: {}'.format(self.input_file))
            self.preprocess()
        else:
            if is_forced:
                print('Forcing to read a file: {}'.format(self.input_file))
                self.preprocess()
            else:
                print('Lpading a file: {}'.format(self.input_file))
                self.load_preprocessed(self.vocab_file, self.tensor_file)
        self.create_batches()
        self.reset_batch_pointer()
    def preprocess(self):
        with codecs.open(self.input_file, 'r', encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        # -x[1] : drives it to sord the counter by a decreasing order based on its' values not key because key is a character
        # Conversely +x[1] : would sort it by ascending order
        count_pairs = sorted(counter.items(), key=lambda  x: -x[1])

        # * let it to split chars and its counters into two different lists
        self.chars, _ = zip(*count_pairs)

        self.vocab_size = len(self.chars)

        # to assign indexes of each char and keep them in a dict called vocab
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        # save chars as pickle
        with open(self.vocab_file, 'wb' ) as f_out:
            cPickle.dump(self.chars, f_out)

        # encode data with a value of each character.
        encoded_data = list(map(self.vocab.get, data))

        #keep them in a tensor
        self.tensor = np.array(encoded_data)

        #save this tensor
        np.save(self.tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file + '.npy')
    def reset_batch_pointer(self):
        self.pointer = 0

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # when data/tensor is too small,
        # let's giv them a better error message

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size_small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]


        # -1 represents the virtual column so that it can reshape the xdata that has a row number of self.batch_size,
        # column size is the remaining part of xdata. Let us say, xdata shape is [100, 100], and batch size = 10,
        # xdata.reshape(self.batch_size, -1) is going to give us an array, that has 10 array and each of array has 1000 elemnts
        # in addition to that, split data function is splitting this reshaped array into the sub arrays along axis=1 so the each element of
        # each array will be stacked together having these new arrays along same axis.
        # Long story short, it has n batches, each of this batches has m data and each of this m data has l elements
        # so we want to process 1 batch at a time which means that one m x l at one time
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, axis=1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, axis=1)
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x,y

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.vocab[word])
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.chars[index])
        return ''.join(words)