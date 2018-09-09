import tensorflow as tf
import os
import sys

from numpy.ma import in1d
from prompt_toolkit.key_binding.bindings.named_commands import accept_line

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import get_variable_name_as_str

def orchestrate_model(questions, params):

    scope = params.model["active_model"]
    with tf.variable_scope(scope):
        tf.logging.info("Question shape: {}...".format(questions))
        output = eval(scope)(questions, params)
        tf.contrib.layers.summarize_activation(output)
        normalized_output = tf.nn.l2_normalize(output, axis=1)
        tf.contrib.layers.summarize_activation(normalized_output)
    return normalized_output

def model_1(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))

    conf = params.model["model_1"]

    with tf.variable_scope('fc'):
        fc_linear = tf.contrib.layers.fully_connected(
            input,
            conf['embedding_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope='linear'
        )

        output = tf.add(fc_linear * conf['scaling_factor'], input, name='linear_add')

    return output

def model_2(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_2.__name__))

    conf = params.model["model_2"]
    _in_out = input
    for i, block_conf in enumerate(conf):
        _in_out = residual_block(_in_out, block_conf, "res_block_{}".format(i))
    return _in_out


# def model_2_vaiation(input, params):
#
#     # Define the model
#     tf.logging.info("Creating the {}...".format(model_3.__name__))
#
#     conf = params.model["model_3"]
#     _in_out = input
#     for i, block_conf in enumerate(conf):
#         _in_out = residual_block(_in_out, block_conf, "res_block_{}".format(i), 2) #TODO: (parameter = 2 = Number of activation layer) can be defined in params.json but this model was not very useful.
#     return _in_out

def model_3(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_3.__name__))

    conf = params.model["model_3"][0]

    with tf.variable_scope('CNN'):
        dropout_emb = questions = tf.contrib.layers.embed_sequence(
            input, params.files['questions_vocab_size'], params.files['pre_trained_files']['embedding_dim'],
            initializer=params.model['conv_embedding_initializer'])

        # dropout_emb = tf.layers.dropout(inputs=questions,
        #                                rate=conf['keep_prob'],
        #                                training=True)
        conv = tf.layers.conv1d(
            inputs=dropout_emb,
            filters=conf['number_of_filters'],
            kernel_size=conf['kernel_size'],
            padding="same",
            activation=tf.nn.relu)



        # Global Max Pooling
        pool = tf.reduce_max(input_tensor=conv, axis=1)

        dropout_hidden = hidden = tf.layers.dense(inputs=pool, units=conf['embedding_dim'], activation=tf.nn.relu)

        # dropout_hidden = tf.layers.dropout(inputs=hidden,
        #                                    rate=conf['keep_prob'],
        #                                    training=True)

        output = tf.layers.dense(inputs=dropout_hidden, units=conf['final_unit'])


    return output


def residual_block(input, conf, scope, num_of_activation_layer=1):
    with tf.variable_scope(scope):
        _input = input
        for i in range(num_of_activation_layer):
            fc_relu = tf.contrib.layers.fully_connected(
                _input,
                conf['fc_relu_embedding_dim'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                    stddev=0.1),
                weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
                biases_initializer=tf.zeros_initializer(),
                trainable=True,
                scope="{}_{}_{}".format(scope,'relu', i)
            )
            dropout = tf.contrib.layers.dropout(fc_relu, conf['keep_prob'], scope="{}_{}".format(scope,'dropout'))
            _input = dropout

        fc_linear = tf.contrib.layers.fully_connected(
            _input,
            conf['fc_non_embedding_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="{}_{}".format(scope,'linear')
        )

        output = tf.add(fc_linear * conf['scaling_factor'], input, name="{}_{}".format(scope,'add'))

    return output
