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