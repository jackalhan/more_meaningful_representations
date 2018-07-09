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

    conf_ = params.model["model_3"]
    with tf.variable_scope('fc'):
        scope = 'fc'
        input_ = tf.reshape(input, (-1, 32, 32))
        x_ = conv_and_res_block(input_, conf_, 32, stage=1)
        x_ = conv_and_res_block(x_, conf_, 64, stage=2)
        x_ = conv_and_res_block(x_, conf_, 128, stage=3)
        x_ = conv_and_res_block(x_, conf_, 256, stage=4)
        x_ = conv_and_res_block(x_, conf_, 512, stage=5)
        # fc_relu = tf.contrib.layers.De(
        #     x_,
        #     conf_['fc_relu_embedding_dim'],
        #     activation_fn=tf.nn.relu,
        #     weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
        #                                                         stddev=0.1),
        #     weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
        #     biases_initializer=tf.zeros_initializer(),
        #     trainable=True,
        #     scope="{}_{}".format(scope, 'relu')
        # )
        flat = tf.reshape(x_, (-1, 16 * 32))
        output = tf.layers.dense(flat ,  conf_['fc_relu_embedding_dim'], name='affine')


        output = tf.add(output * conf_['scaling_factor'], input, name="{}_{}".format(scope, 'add'))

    return output

def conv_and_res_block(input, conf, filters, stage):
    conv_name = 'conv{}-s'.format(filters)
    o = tf.layers.conv1d(inputs=input,
                             filters=filters,
                             kernel_size=5,
                             strides=2,
                             padding='same',
                             #activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                                stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
                             trainable=True,
                             name=conv_name
                             )
    #o = tf.nn.batch_normalization(o, name=conv_name + '_bn')
    #o = tf.layers.max_pooling1d(inputs=o, pool_size=2, strides=2, padding='same', name=conv_name + '_max_pool')
    for i in range(10):
        o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i, conf=conf)
    return o

def identity_block(input, kernel_size, filters, stage, block, conf):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = tf.layers.conv1d(inputs=input,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same',
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                            stddev=0.1),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
                         trainable=True,
                         name=conv_name_base + '_2a'
                         )
    x = tf.layers.conv1d(inputs=x,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same',
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
                                                                            stddev=0.1),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
                         trainable=True,
                         name=conv_name_base + '_2b'
                         )
    #x = tf.add(x, input, name=conv_name_base + '_add')
    x = tf.add(x, input, name=conv_name_base + '_add')
    return x


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
