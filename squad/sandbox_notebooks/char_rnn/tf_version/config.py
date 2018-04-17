import os
import tensorflow as tf
flags = tf.app.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags.DEFINE_string("data_dir", 'data/tinyshakespare', 'data directory containing data file')
flags.DEFINE_string("data_file", 'input.txt', 'data file name in the folder specified in data_dir')
flags.DEFINE_string("save_dir", 'data/tinyshakespare/save', 'directory to store checkpointed models')
flags.DEFINE_string("log_dir", 'data/tinyshakespare/logs', 'directory to store tensorboard logs')
flags.DEFINE_bool("is_forced", True, 'Whether you want to force to reprocess data or not if data is already processed')
flags.DEFINE_bool("is_one_hot_embedding", False, 'Whether you want to embed the input using one hot encoder')

flags.DEFINE_integer("hidden_size", 128, 'size of RNN hidden state')
flags.DEFINE_integer("num_layers", 4, 'number of layers in the RNN')
flags.DEFINE_string('model_type', 'gru', 'rnn, gru, lstm, nas')
flags.DEFINE_string('activation', 'relu', 'tanh, relu')
flags.DEFINE_integer('batch_size', 50, 'minibatch size')
flags.DEFINE_integer('seq_length', 50, 'sequence length of the model')
flags.DEFINE_integer('num_epochs', 100, 'num of epochs')
flags.DEFINE_integer('save_every', 1000, 'save frequency')
flags.DEFINE_integer('log_every', 10, 'log frequency')
flags.DEFINE_integer('grad_clip', 5, 'clip gradients at this value to prevent gradient explosion')
flags.DEFINE_string('optimizer', 'adam', 'rmsprop, adam, adadelta, adagrad')

flags.DEFINE_float('learning_rate', 0.0008, 'learning rate')
flags.DEFINE_float('decay_rate', 0.97, 'decay rate')

flags.DEFINE_float('keep_prop_output_layer', 0.4, 'probability of keeping weights in the output layer')
flags.DEFINE_float('keep_prop_input_layer', 0.4, 'probability of keeping weights in the input layer')

# flags.DEFINE_string('init_from', 'data/tinyshakespare/save', """continue training from saved model at this path. Path must contain files saved by previous training process:
#                                 'config.pkl'        : configuration;
#                                 'chars_vocab.pkl'   : vocabulary definitions;
#                                 'checkpoint'        : paths to model file(s) (created by tf).
#                                                       Note: this file contains absolute paths, be careful when moving files around;
#                                 'model.ckpt-*'      : file(s) with model definition (created by tf)
#                             """)
flags.DEFINE_string('init_from', None, """continue training from saved model at this path. Path must contain files saved by previous training process:
                                'config.pkl'        : configuration;
                                'chars_vocab.pkl'   : vocabulary definitions;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
flags.DEFINE_string('start_string', 'The', 'use this string to start generating')
flags.DEFINE_integer("max_gen_length", 1000, 'the length of the max character generation')
FLAGS = tf.app.flags.FLAGS