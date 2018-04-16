from __future__ import print_function
import argparse
import time
import os
from six.moves import cPickle
from squad.sandbox_notebooks.char_rnn.tf_version.utils import TextLoader
from squad.sandbox_notebooks.char_rnn.tf_version.model import Model
import squad.sandbox_notebooks.char_rnn.tf_version.config as config

# def main():
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--data_dir',
#                         type=str,
#                         default='data/tinyshakespare',
#                         help='data directory containing data file')
#
#     parser.add_argument('--data_file',
#                         type=str,
#                         default='input.txt',
#                         help='data file name in the folder specified in data_dir')
#
#     parser.add_argument('--is_forced',
#                         type=bool,
#                         default=True,
#                         help='Whether you want to force to reprocess data or not if data is already processed')
#
#     parser.add_argument('--is_one_hot_embedding',
#                         type=bool,
#                         default=False,
#                         help='Whether you want to embed the input using one hot encoder')
#
#     parser.add_argument('--save_dir',
#                         type=str,
#                         default='data/tinyshakespare/save',
#                         help='directory to store checkpointed models')
#
#     parser.add_argument('--log_dir',
#                         type=str,
#                         default='data/tinyshakespare/logs',
#                         help='directory to store tensorboard logs')
#
#     parser.add_argument('--hidden_size',
#                         type=int,
#                         default=128,
#                         help='size of RNN hidden state')
#
#     parser.add_argument('--num_layers',
#                         type=int,
#                         default=2,
#                         help='number of layers in the RNN')
#
#     parser.add_argument('--model_type',
#                         type=str,
#                         default='lstm',
#                         help='rnn, gru, lstm, nas')
#
#     parser.add_argument('--activation',
#                         type=str,
#                         default='relu',
#                         help='tanh,relu')
#
#     parser.add_argument('--batch_size',
#                         type=int,
#                         default=50,
#                         help='minibatch size')
#
#     parser.add_argument('--seq_length',
#                         type=int,
#                         default=50,
#                         help='sequence length of the model')
#
#     parser.add_argument('--num_epochs',
#                         type=int,
#                         default=50,
#                         help='num of epochs')
#
#     parser.add_argument('--save_every',
#                         type=int,
#                         default=1000,
#                         help='save frequency')
#
#
#     parser.add_argument('--log_every',
#                         type=int,
#                         default=10,
#                         help='log frequency')
#
#     parser.add_argument('--grad_clip',
#                         type=float,
#                         default=5,
#                         help='clip gradients at this value to prevent gradient explosion')
#
#     parser.add_argument('--optimizer',
#                         type=str,
#                         default='adam',
#                         help='rmsprop, adam, adadelta, adagrad')
#
#     parser.add_argument('--learning_rate',
#                         type=float,
#                         default=0.0002,
#                         help='learning rate')
#
#     parser.add_argument('--decay_rate',
#                         type=float,
#                         default=0.97,
#                         help='decay rate')
#
#     parser.add_argument('--keep_prop_output_layer',
#                         type=float,
#                         default=1.0,
#                         help='probability of keeping weights in the output layer')
#
#     parser.add_argument('--keep_prop_input_layer',
#                         type=float,
#                         default=1.0,
#                         help='probability of keeping weights in the input layer')
#
#     parser.add_argument('--init_from', type=str, default=None,
#                         help="""continue training from saved model at this path. Path must contain files saved by previous training process:
#                                 'config.pkl'        : configuration;
#                                 'chars_vocab.pkl'   : vocabulary definitions;
#                                 'checkpoint'        : paths to model file(s) (created by tf).
#                                                       Note: this file contains absolute paths, be careful when moving files around;
#                                 'model.ckpt-*'      : file(s) with model definition (created by tf)
#                             """)
#
#     args = parser.parse_args()
#     train(args)

def train(config):
    data_loader = TextLoader(args.data_dir, args.data_file, args.batch_size, args.seq_length, args.is_forced)
    args.vocab_size = data_loader.vocab_size

    #check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " {} must be a path".format(args.init_from)
        assert os.path.isfile(
            os.path.join(args.init_from, "{}_config.pkl".format(data_loader.data_file_pre))), "{}_config.pkl file does not exist in path {}".format(data_loader.data_file_pre, args.init_from)
        assert os.path.isfile(os.path.join(args.init_from,
                                           "{}_chars_vocab.pkl".format(data_loader.data_file_pre))), "{}_chars_vocab.pkl.pkl file does not exist in path {}".format(data_loader.data_file_pre, args.init_from)
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, "{}_config.pkl".format(data_loader.data_file_pre)), 'rb') as f:
            saved_model_args = cPickle.load(f)

        need_be_same = ["model_type", "hidden_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], 'Command line arg and saved model arg have inconsistency on {}' .format(checkme)

        # open saved vocab/dict and check if vocabs/dict are compatible
        with open(os.path.join(args.init_from,
                                           "{}_chars_vocab.pkl".format(data_loader.data_file_pre)), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, 'Data and loaded model have inconsistency on character set!'
        assert saved_vocab == data_loader.vocab, 'Data and loaded model have inconsistency on dictionary mappings!'

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "{}_config.pkl".format(data_loader.data_file_pre)), 'wb') as f:
        cPickle.dump(args, f)

    with open(os.path.join(args.save_dir,
                                           "{}_chars_vocab.pkl".format(data_loader.data_file_pre)), 'wb') as f:
        cPickle.dump((data_loader.chars,data_loader.vocab), f)
    with tf.Session() as sess:
        model = Model(args)
        # INSTRUMENT FOR TENSORBOARD
        # Merges all summaries collected in the default graph.
        # summaries = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(
        #     os.path.join(args.log_dir,  time.strftime("%Y-%m-%d-%H-%M-%S"))
        # )
        # writer.add_graph(sess.graph)

        # define training procedure

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # RESTORE THE MODEL
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            model_lr = tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e))
            sess.run([model_lr])
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_cell_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data:x,
                        model.target_data:y,
                        model.initial_cell_state:state}
                '''
                c: cell state / cell memory
                h: hidden state that is an outut of the cell so that next time step can use it
                '''
                # for i, (c, h) in enumerate(model.initial_cell_state):
                #     feed[c] = state[i].c
                #     feed[h] = state[i].h

                #instribument for tensorboard
                _, step, new_state, loss = sess.run([
                    model.train_op, model.global_step, model.final_state, model.loss
                ], feed)
                end = time.time()
                current_step = tf.train.global_step(sess, model.global_step)

                if step % args.log_every == 0:
                    print('step: {}/{}... '.format(int(step), args.num_epochs * data_loader.num_batches),
                          'loss: {:.4f}... '.format(loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if current_step % args.save_every == 0:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=current_step)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    config.tf.app.run()
    my_config = config.flags.FLAGS
    train(my_config)