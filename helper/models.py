import tensorflow as tf


def model_1(questions, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))
    fc1 = tf.contrib.layers.fully_connected(
        questions,
        int(params.embedding_dim),
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(seed=params.seed, stddev=0.1),
        weights_regularizer=tf.contrib.layers.l2_regularizer(params.l2_regularizer),
        biases_initializer=tf.zeros_initializer(),
        trainable=True,
        )
    tf.contrib.layers.summarize_activation(fc1)
    _questions = tf.add(fc1 * params.scaling_factor, questions, name='add')
    _trained_question_embeddings = tf.nn.l2_normalize(_questions, name='normalize', axis=1)
    return _trained_question_embeddings

def model_2(questions, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_2.__name__))
    residual_block_1 = {"fc_relu_dim": params.embedding_dim,
                        "fc_relu_seed": params.seed,
                        "fc_relu_weight_decay": params.l2_regularizer,
                        "keep_prob":params.keep_prob,
                        "fc_linear_dim": params.embedding_dim,
                        "fc_linear_seed": params.seed,
                        "fc_linear_weight_decay": params.l2_regularizer,
                        "scaling_factor":params.scaling_factor,
                        "name":"residual_block_1"
                        },
    residual_block_2 = {"fc_relu_dim": params.embedding_dim,
                        "fc_relu_seed": params.seed,
                        "fc_relu_weight_decay": params.l2_regularizer,
                        "keep_prob": params.keep_prob,
                        "fc_linear_dim": params.embedding_dim,
                        "fc_linear_seed": params.seed,
                        "fc_linear_weight_decay": params.l2_regularizer,
                        "scaling_factor": params.scaling_factor,
                        "name": "residual_block_2"
                        },
    blocks = [residual_block_1]
    _questions = questions
    for block in blocks[0]:
        _questions = residual_block(_questions, block)
    _trained_question_embeddings = tf.nn.l2_normalize(_questions, name='normalize', axis=1)
    return _trained_question_embeddings

def residual_block(input, block_conf):

    with tf.name_scope(block_conf['name']) as scope:
        fc_relu = tf.contrib.layers.fully_connected(
            input,
            block_conf['fc_relu_dim'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(seed=block_conf['fc_relu_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(block_conf['fc_relu_weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope='{}_fc_relu'.format(scope)
        )

        drop_out = tf.nn.dropout(fc_relu, block_conf['keep_prob'],name="{}_dropout".format(scope))  # DROP-OUT here
        fc_linear = tf.contrib.layers.fully_connected(
            drop_out,
            block_conf['fc_linear_dim'],
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=block_conf['fc_linear_seed'],
                                                                stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(block_conf['fc_linear_weight_decay']),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope='{}_fc_linear'.format(scope)
        )
        tf.contrib.layers.summarize_activation(fc_relu)
        tf.contrib.layers.summarize_activation(drop_out)
        tf.contrib.layers.summarize_activation(fc_linear)
        output = tf.add(fc_linear * block_conf['scaling_factor'], input, name='{}_add'.format(scope))

    return output
