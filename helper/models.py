import tensorflow as tf


def model_1(questions, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))
    with tf.name_scope('fc') as scope:
        fc1 = tf.contrib.layers.fully_connected(
            questions,
            int(params.embedding_dim),
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(seed=params.seed, stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(params.l2_regularizer),
            biases_initializer=tf.constant_initializer(0),
            scope="{}_fc1".format(scope),
            trainable=True
        )
        data_ = tf.add(fc1 * params.scaling_factor, questions, name='add')
        out = tf.nn.l2_normalize(data_, name='out', axis=1)
    return out, fc1