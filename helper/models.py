import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import get_variable_name_as_str

def orchestrate_model(questions, params):

    scope = params.model_name
    with tf.variable_scope(scope):
        output = eval(params.model_name)(questions, params)
        tf.contrib.layers.summarize_activation(output)
        normalized_output = tf.nn.l2_normalize(output, axis=1)
        tf.contrib.layers.summarize_activation(normalized_output)
    return normalized_output

def model_1(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))


    conf = {"fc_linear_dim": params.embedding_dim,
            "fc_linear_seed": params.seed,
            "fc_linear_weight_decay": params.l2_regularizer,
            "scaling_factor": params.scaling_factor,
            }
    with tf.variable_scope('fc'):
        fc_linear = tf.contrib.layers.fully_connected(
            input,
            conf['fc_linear_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['fc_linear_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['fc_linear_weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope='linear'
        )

        output = tf.add(fc_linear * conf['scaling_factor'], input, name='linear_add')

    return output

def model_2(input, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_2.__name__))
    residual_block_same_cells = {"fc_relu_dim": params.embedding_dim,
                        "fc_relu_seed": params.seed,
                        "fc_relu_weight_decay": params.l2_regularizer,
                        "keep_prob":params.keep_prob,
                        "fc_linear_dim": params.embedding_dim,
                        "fc_linear_seed": params.seed,
                        "fc_linear_weight_decay": params.l2_regularizer,
                        "scaling_factor":params.scaling_factor
                        },
    residual_block_large_cells = {"fc_relu_dim": 2048,
                        "fc_relu_seed":  params.seed,
                        "fc_relu_weight_decay": params.l2_regularizer,
                        "keep_prob": params.keep_prob,
                        "fc_linear_dim": params.embedding_dim,
                        "fc_linear_seed": params.seed,
                        "fc_linear_weight_decay": params.l2_regularizer,
                        "scaling_factor": params.scaling_factor
                        },
    blocks = [residual_block_large_cells,residual_block_same_cells]
    _in_out = input
    for i, block_conf in enumerate(blocks):
        _in_out = residual_block(_in_out, block_conf[0], "res_block_{}".format(i))
    return _in_out

def residual_block(input, conf, scope):
    with tf.variable_scope(scope):
        fc_relu = tf.contrib.layers.fully_connected(
            input,
            conf['fc_relu_dim'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['fc_relu_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['fc_relu_weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="{}_{}".format(scope,'relu')
        )

        dropout = tf.contrib.layers.dropout(fc_relu, conf['keep_prob'], scope="{}_{}".format(scope,'dropout'))
        fc_linear = tf.contrib.layers.fully_connected(
            dropout,
            conf['fc_linear_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=conf['fc_linear_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(conf['fc_linear_weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="{}_{}".format(scope,'linear')
        )

        output = tf.add(fc_linear * conf['scaling_factor'], input, name="{}_{}".format(scope,'add'))

    return output
