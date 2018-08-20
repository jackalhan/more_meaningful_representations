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
    _in_out = input
    for i, block_conf in enumerate(conf):
        _in_out = residual_block(_in_out, block_conf, "res_block_{}".format(i), 2) #TODO: (parameter = 2 = Number of activation layer) can be defined in params.json but this model was not very useful.
    return _in_out

def model_4(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_4.__name__))

    conf = params.model["model_4"]

    with tf.variable_scope('CNN_Layer_1_bigram'):
        # Apply Convolution filtering on input sequence.
        conv1_bigram = tf.layers.conv1d(
            input,
            filters=conf['embedding_dim'],
            kernel_size=2,
            padding='VALID',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1_bigram = tf.layers.max_pooling1d(conv1_bigram)
        # Transpose matrix so that n_filters from convolution becomes width.
        #pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer_1_trigram'):
        # Apply Convolution filtering on input sequence.
        conv1_trigram = tf.layers.conv1d(
            input,
            filters=conf['embedding_dim'],
            kernel_size=2,
            padding='VALID',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1_trigram = tf.layers.max_pooling1d(conv1_trigram)
    with tf.variable_scope('CNN_Layer_1_fourgram'):
        # Apply Convolution filtering on input sequence.
        conv1_fourgram = tf.layers.conv1d(
            input,
            filters=conf['embedding_dim'],
            kernel_size=2,
            padding='VALID',
            # Add a ReLU for non linearity.
            activation=tf.nn.relu)
        # Max pooling across output of Convolution+Relu.
        pool1_fourgram = tf.layers.max_pooling1d(conv1_fourgram)
    #
    # with tf.variable_scope('CNN_Layer2'):
    #     # Second level of convolution filtering.
    #     conv2 = tf.layers.conv2d(
    #         pool1,
    #         filters=N_FILTERS,
    #         kernel_size=FILTER_SHAPE2,
    #         padding='VALID')
    #     # Max across each filter to get useful features for classification.
    #     pool2 = tf.squeeze(tf.reduce_max(conv2, 1), axis=[1])
    merged = tf.concat([pool1_bigram, pool1_trigram, pool1_fourgram], axis=1)
    dense = tf.layers.Dense(merged, conf['embedding_dim'], activation=tf.nn.relu)

    return dense


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
# def model_3(input, params):
#     # Define the model
#     tf.logging.info("Creating the {}...".format(model_3.__name__))
#
#     conf_ = params.model["model_3"]
#     with tf.variable_scope('fc'):
#         scope = 'fc'
#         input_ = tf.reshape(input, (-1, 32, 32)) #best 32 32 with a sf of 0.2
#         x_ = conv_and_res_block(input_, conf_, 4    , stage=1) #best 4
#         #x_ = conv_and_res_block(x_, conf_, 128, stage=3)
#         # # x_ = conv_and_res_block(x_, conf_, 256, stage=4)
#         # x_ = conv_and_res_block(x_, conf_, 512, stage=5)
#         #x_ = conv_and_res_block(x_, conf_, 1024, stage=6)
#
#         # fc_relu = tf.contrib.layers.De(
#         #     x_,
#         #     conf_['fc_relu_embedding_dim'],
#         #     activation_fn=tf.nn.relu,
#         #     weights_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
#         #                                                         stddev=0.1),
#         #     weights_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
#         #     biases_initializer=tf.zeros_initializer(),
#         #     trainable=True,
#         #     scope="{}_{}".format(scope, 'relu')
#         # )
#         flat = tf.reshape(x_, (-1, 2 * 32)) #best 2,32
#         output = tf.layers.dense(flat ,  conf_['fc_relu_embedding_dim'], name='affine')
#
#
#         output = tf.add(output * conf_['scaling_factor'], input, name="{}_{}".format(scope, 'add'))
#
#     return output

# def conv_and_res_block(input, conf, filters, stage):
#     conv_name = 'conv{}-s'.format(filters)
#     o = tf.layers.conv1d(inputs=input,
#                              filters=filters,
#                              kernel_size=5,
#                              strides=2,
#                              padding='same',
#                              #activation=tf.nn.relu,
#                              kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
#                                                                                 stddev=0.1),
#                              kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
#                              trainable=True,
#                              name=conv_name
#                              )
#     #o = tf.nn.batch_normalization(o, name=conv_name + '_bn')
#     #o = tf.layers.max_pooling1d(inputs=o, pool_size=2, strides=2, padding='same', name=conv_name + '_max_pool')
#     for i in range(10):
#         o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i, conf=conf)
#     return o
#
# def identity_block(input, kernel_size, filters, stage, block, conf):
#     conv_name_base = 'res{}_{}_branch'.format(stage, block)
#
#     x = tf.layers.conv1d(inputs=input,
#                          filters=filters,
#                          kernel_size=kernel_size,
#                          strides=1,
#                          padding='same',
#                          activation=None,
#                          kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
#                                                                             stddev=0.1),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
#                          trainable=True,
#                          name=conv_name_base + '_2a'
#                          )
#     x = tf.layers.conv1d(inputs=x,
#                          filters=filters,
#                          kernel_size=kernel_size,
#                          strides=1,
#                          padding='same',
#                          activation=None,
#                          kernel_initializer=tf.truncated_normal_initializer(seed=conf['initializer_seed'],
#                                                                             stddev=0.1),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(conf['weight_decay']),
#                          trainable=True,
#                          name=conv_name_base + '_2b'
#                          )
#     #x = tf.add(x, input, name=conv_name_base + '_add')
#     x = tf.add(x, input, name=conv_name_base + '_add')
#     return x


