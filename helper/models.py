import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import get_variable_name_as_str

def orchestrate_model(questions, params):

    scope = params.model["active_model"]
    with tf.variable_scope(scope):
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

def model_3(input, params):
    # Define the model
    tf.logging.info("Creating the {}...".format(model_3.__name__))

    conf = params.model["model_3"]

    with tf.variable_scope('fc'):
        input_ = tf.reshape(input, (-1, 128, 8))
        conv1 = tf.layers.conv1d(inputs=input_, filters=16, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=32, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

        conv4= tf.layers.conv1d(inputs=max_pool_3, filters=128, kernel_size=2, strides=1,
                                 padding='same', activation=tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        flat = tf.reshape(max_pool_4, (-1, 32 * 32))

        fc_linear = tf.contrib.layers.fully_connected(
            flat,
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

def residual_block(input, conf, scope):
    with tf.variable_scope(scope):
        fc_relu = tf.contrib.layers.fully_connected(
            input,
            conf['fc_relu_embedding_dim'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="{}_{}".format(scope,'relu')
        )

        dropout = tf.contrib.layers.dropout(fc_relu, conf['keep_prob'], scope="{}_{}".format(scope,'dropout'))
        fc_linear = tf.contrib.layers.fully_connected(
            dropout,
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
