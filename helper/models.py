import tensorflow as tf
import os
import sys

from numpy.ma import in1d
from prompt_toolkit.key_binding.bindings.named_commands import accept_line

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import get_variable_name_as_str

def orchestrate_model(source, params):

    scope = params.model["active_model"]
    with tf.variable_scope(scope):
        tf.logging.info("Source shape: {}...".format(source))
        output = eval(scope)(source, params)
        tf.contrib.layers.summarize_activation(output)
        normalized_output = tf.nn.l2_normalize(output, axis=1)
        tf.contrib.layers.summarize_activation(normalized_output)
    return normalized_output

def model_1(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))

    conf = params.model["model_1"]

    source = input['source_embeddings']
    with tf.variable_scope('fc'):
        fc_linear = tf.contrib.layers.fully_connected(
            source,
            conf['embedding_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope='linear'
        )

        output = tf.add(fc_linear * conf['scaling_factor'], source, name='linear_add')

    return output

def model_2(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_2.__name__))

    conf = params.model["model_2"]
    source = input['source_embeddings']
    _in_out = source
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
    baseline_source_embeddings = input['baseline_source_embeddings']
    source_embeddings = input['source_embeddings']
    with tf.variable_scope('CNN'):
        embedding_layer = tf.contrib.layers.embed_sequence(
            source_embeddings, params.files['vocab_size'], params.files['pre_trained_files']['embedding_dim'],
            initializer=params.model['conv_embedding_initializer'])

        conv1 = tf.layers.conv1d(embedding_layer, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        conv2 = tf.layers.conv1d(conv1, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        conv3 = tf.layers.conv1d(conv2, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        min_avg_pooling = tf.reduce_min(conv3, axis=1)

        # dropout_hidden = tf.layers.dropout(inputs=min_avg_pooling, rate=conf['keep_prob'])
        #
        # dense_output = tf.layers.dense(inputs=dropout_hidden, units=conf['final_unit'])

        #
        # pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
        #
        # conv3 = tf.layers.conv1d(pool2, 1024, kernel_size=3, padding="same", activation=tf.nn.relu)
        # pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)

        # conv4 = tf.layers.conv1d(pool3, 4096, kernel_size=3, padding="same", activation=tf.nn.relu)
        # pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2)
        #
        # conv4 = tf.layers.conv1d(pool3, 4096, kernel_size=3, padding="same", activation=tf.nn.relu)
        # pool4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2)
        #
        # pool4_flat = tf.reshape(pool4, [-1, 8 * 4096])
        #
        # dropout_hidden = tf.layers.dropout(inputs=pool4_flat, rate=conf['keep_prob'])
        # dense_output = tf.layers.dense(dropout_hidden, conf['final_unit'])
        #net = tf.layers.dense(net, self.num_classes)


        # dropout_emb = tf.layers.dropout(inputs=questions,
        #                                rate=conf['keep_prob'],
        #                                training=True)
        # conv = tf.layers.conv1d(
        #     inputs=dropout_emb,
        #     filters=conf['number_of_filters'],
        #     kernel_size=conf['kernel_size'],
        #     padding="same",
        #     activation=tf.nn.relu)
        #
        #
        # # Global Max Pooling
        # pool = tf.reduce_max(input_tensor=conv, axis=1)
        #
        # hidden = tf.layers.dense(inputs=pool, units=conf['embedding_dim'], activation=tf.nn.relu)
        #
        # dropout_hidden = tf.layers.dropout(inputs=hidden,
        #                                    rate=conf['keep_prob'])
        #
        # dense_output = tf.layers.dense(inputs=dropout_hidden, units=conf['final_unit'])

        output = tf.add(min_avg_pooling * conf['scaling_factor'], baseline_source_embeddings)
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
