from __future__ import print_function
import time
import os
import tensorflow as tf
import numpy as np
from six.moves import cPickle
from squad.sandbox_notebooks.char_rnn.tf_version.utils import TextLoader
from squad.sandbox_notebooks.char_rnn.tf_version.model import Model
import squad.sandbox_notebooks.char_rnn.tf_version.config as my_config


def main(_):
    config = my_config.flags.FLAGS
    config.is_forced = False
    data_loader = TextLoader(config.data_dir,
                             config.data_file,
                             config.batch_size,
                             config.seq_length,
                             config.is_forced)
    config.vocab_size = data_loader.vocab_size
    #check compatibility if training is continued from previously saved model
    # check if all necessary files exist
    assert os.path.isdir(config.init_from), " {} must be a path".format(config.init_from)
    assert os.path.isfile(
        os.path.join(config.init_from, "{}_config.pkl".format(data_loader.data_file_pre))), "{}_config.pkl file does not exist in path {}".format(data_loader.data_file_pre, config.init_from)
    assert os.path.isfile(os.path.join(config.init_from,
                                       "{}_chars_vocab.pkl".format(data_loader.data_file_pre))), "{}_chars_vocab.pkl.pkl file does not exist in path {}".format(data_loader.data_file_pre, config.init_from)
    ckpt = tf.train.get_checkpoint_state(config.init_from)
    assert ckpt, "No checkpoint found"
    assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

    # open old config and check if models are compatible
    with open(os.path.join(config.init_from, "{}_config.pkl".format(data_loader.data_file_pre)), 'rb') as f:
        saved_model_args = cPickle.load(f)

    need_be_same = ["model_type", "hidden_size", "num_layers", "seq_length"]
    for checkme in need_be_same:
        assert vars(saved_model_args)['__flags'][checkme] == vars(config)['__flags'][checkme], 'Command line arg and saved model arg have inconsistency on {}' .format(checkme)

    # open saved vocab/dict and check if vocabs/dict are compatible
    with open(os.path.join(config.init_from,
                                       "{}_chars_vocab.pkl".format(data_loader.data_file_pre)), 'rb') as f:
        saved_chars, saved_vocab = cPickle.load(f)
    assert saved_chars==data_loader.chars, 'Data and loaded model have inconsistency on character set!'
    assert saved_vocab == data_loader.vocab, 'Data and loaded model have inconsistency on dictionary mappings!'

    start = data_loader.text_to_arr(config.start_string)
    with tf.Session() as sess:
        model = Model(config,False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # RESTORE THE MODEL
        saver.restore(sess, ckpt.model_checkpoint_path)
        samples = [c for c in start]
        state = sess.run(model.initial_cell_state)
        preds = np.ones((data_loader.vocab_size))
        for c in start:
            preds = char_generation(sess, model, c, state)

        c = pick_top_n(preds, data_loader.vocab_size)
        samples.append(c)
        #generate character until it reaches the specified size
        for i in range(config.max_gen_length):
            preds = char_generation(sess, model, c, state)
            c = pick_top_n(preds, data_loader.vocab_size)
            samples.append(c)
        samples = np.array(samples)
        print(data_loader.arr_to_text(samples))

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
def char_generation(sess, model, top_n_chars, state):
    x = np.zeros((1, 1))
    x[0, 0] = top_n_chars
    feed = {model.input_data: x,
            model.initial_cell_state: state}
    preds, new_state = sess.run([model.probs, model.final_state], feed)
    return preds
if __name__ == '__main__':
    tf.app.run()