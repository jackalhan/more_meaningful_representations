import tensorflow as tf
import os
import sys

from numpy.ma import in1d
#from prompt_toolkit.key_binding.bindings.named_commands import accept_line

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import get_variable_name_as_str

def orchestrate_model(source_embeddings, source_baseline_embeddings,  params):

    scope = params.model["active_model"]
    with tf.variable_scope(scope):
        tf.logging.info("Source shape: {}...".format(source_embeddings))
        output = eval(scope)(source_embeddings, source_baseline_embeddings, params)
        tf.contrib.layers.summarize_activation(output)
    return output

def model_1(source_embeddings, source_baseline_embeddings, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_1.__name__))

    conf = params.model["model_1"]

    source = source_embeddings
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

def model_2(source_embeddings, source_baseline_embeddings, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_2.__name__))

    conf = params.model["model_2"]
    source = source_embeddings
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

def model_3(source_embeddings, source_baseline_embeddings, params):

    # Define the model
    tf.logging.info("Creating the {}...".format(model_3.__name__))

    conf = params.model["model_3"][0]

    with tf.variable_scope('CNN'):
        embedding_layer = tf.contrib.layers.embed_sequence(
            source_embeddings, params.files['vocab_size'], params.files['pre_trained_files']['embedding_dim'],
            initializer=params.model['conv_embedding_initializer'])

        dropout_emb = tf.layers.dropout(inputs=embedding_layer,
                                       rate=conf['keep_prob'])

        conv1 = tf.layers.conv1d(dropout_emb, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        conv2 = tf.layers.conv1d(conv1, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        conv3 = tf.layers.conv1d(conv2, 1024, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)

        avg_pooling = tf.reduce_mean(conv3, axis=1)

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

        output = tf.add(avg_pooling * conf['scaling_factor'], source_baseline_embeddings)
    return output

def model_4(source_embeddings, source_baseline_embeddings, params):
    # Define the model
    tf.logging.info("Creating the {}...".format(model_4.__name__))

    conf = params.model["model_4"][0]

    with tf.variable_scope('WaveNet'):
        embedding_layer = tf.contrib.layers.embed_sequence(
            source_embeddings, params.files['vocab_size'], params.files['pre_trained_files']['embedding_dim'],
            initializer=params.model['conv_embedding_initializer'])

        l1a, l1b = wavenet_block(embedding_layer, embedding_layer, 1024, 3, 1, '1', conf)
        l2a, l2b = wavenet_block(l1a, l1a, 1024, 3, 2, '2', conf)
        l3a, l3b = wavenet_block(l2a, l2a, 1024, 3, 4, '3' , conf)
        l4a, l4b = wavenet_block(l3a, l3a, 1024, 3, 8, '4', conf)
        l5a, l5b = wavenet_block(l4a, l4a, 1024, 3, 16, '5' , conf)
        l6 = tf.keras.layers.Concatenate(axis=-1)([l1b, l2b, l3b, l4b, l5b])
        #l7 = tf.keras.layers.Lambda(tf.nn.relu)(l6)
        l8 = tf.layers.conv1d(l6, 1024, 1, padding="same")
        l9 = tf.layers.conv1d(l6, 1024, 1, padding="same")
        # log_amp_valid =tf.keras.layers.Cropping1D((steps_leadin, 0), name='crop_a')(log_amp)
        # phase_shift_valid =tf.keras.layers.Cropping1D((steps_leadin, 0), name='crop_p')(phase_shift)
        out = tf.keras.layers.Concatenate(axis=-1)([l8, l9])
        avg_pooling = tf.reduce_mean(out, axis=1)
        dense_output = tf.layers.dense(inputs=avg_pooling, units=conf['final_unit'])
    return dense_output

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

def wavenet_block(input, embeddings, channels, kernel_size, dilation_rate, name, conf):

    filter_out = tf.layers.conv1d(embeddings, channels,
                                  kernel_size=kernel_size,
                                  strides=1, dilation_rate=dilation_rate,
                                  padding="valid", use_bias=True,
                                  activation=tf.nn.tanh, name='filter_' + name)

    gate_out = tf.layers.conv1d(embeddings, channels,
                                  kernel_size=kernel_size,
                                  strides=1, dilation_rate=dilation_rate,
                                  padding="valid", use_bias=True,
                                  activation=tf.nn.sigmoid, name='gate_' + name)

    mult = tf.multiply(filter_out, gate_out, name='mult_' + name)

    mult_padded = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0))(mult)

    transformed = tf.layers.conv1d(mult_padded, channels,
                                  kernel_size=1,
                                  padding="same", use_bias=True,
                                  activation=None, name='trans_' + name)

    skip_out = tf.layers.conv1d(mult_padded, channels,
                                   kernel_size=1,
                                   padding="same", use_bias=True,
                                   activation=tf.nn.relu, name='skip_' + name)

    return tf.add(transformed * conf['scaling_factor'], input, name="resid_" + name), skip_out



# def model_mel_to_spec(input_shape=(steps_total, mel_bins)):
#     # mel = keras.layers.Input(shape=input_shape, name='MelInput')
#     # mel = tf.keras.layers.Input(shape=input_shape, name='MelInput')
#     # mel = tf.keras.layers.Input(batch_size=batch_size, shape=input_shape, name='MelInput')
#     # mel = keras.layers.Input(batch_size=batch_size, shape=input_shape, name='MelInput')
#     # mel = keras.layers.Input(shape=input_shape, name='MelInput',
#     #                        _batch_input_shape = (batch_size, steps_total, mel_bins))
#
#     # mel = keras.layers.Input(batch_shape=(batch_size, steps_total, mel_bins), name='MelInput')
#     # mel._batch_input_shape = (batch_size, steps_total, mel_bins)
#     # mel = keras.layers.InputLayer(input_shape=input_shape, name='MelInput')
#
#     # mel_floored = K.maximum(0.00001, mel)
#     # mel_log     = K.log(mel_floored)  # This is (batch, T. channels)
#
#     # mel_log = log_amplitude_with_minimum(mel)
#
#     mel_log = keras.layers.Input(shape=input_shape, name='MelInput')
#     phase0 = keras.layers.Input(shape=input_shape, name='Phase0')  # Unused
#
#     x = keras.layers.BatchNormalization()(mel_log)
#
#     # 'Resize' to make everything 'io_channels' big at the layer interfaces
#     x = s0 = keras.layers.Conv1D(io_channels, 1,
#                                  padding='same', use_bias=True,
#                                  activation='linear', name='mel_log_expanded')(x)
#
#     x, s1 = wavenet_layer(io_channels, hidden_channels * 1, 3, 1, '1')(x)
#     x, s2 = wavenet_layer(io_channels, hidden_channels * 1, 3, 2, '2')(x)
#     x, s3 = wavenet_layer(io_channels, hidden_channels * 1, 3, 4, '3')(x)
#     x, s4 = wavenet_layer(io_channels, hidden_channels * 1, 3, 8, '4')(x)
#     _, s5 = wavenet_layer(io_channels, hidden_channels * 1, 3, 16, '5')(x)  # Total footprint is ~64 0.75secs
#     # x is now irrelevant
#
#     # skip_overall = keras.layers.Concatenate( axis=-1 )( [s0,s1,s2,s3,s4,s5] )
#     skip_overall = keras.layers.Concatenate(axis=-1)([s0, s1])
#
#     log_amp = keras.layers.Conv1D(spectra_bins, 1, padding='same',
#                                   activation='linear', name='log_amp')(skip_overall)
#     phase_shift = keras.layers.Conv1D(spectra_bins, 1, padding='same',
#                                       activation='linear', name='phase_shift')(skip_overall)
#
#     # return keras.models.Model(inputs=[mel], outputs=[log_amp, phase])
#
#     # amp = K.exp(log_amp)
#     # amp = keras.layers.Lambda( lambda x: K.exp(x), name='amp')(log_amp)
#     # return keras.models.Model(inputs=[mel], outputs=[log_amp, phase])
#
#     # spec_real = keras.layers.Multiply()( [amp, K.cos(phase)] )
#     # spec_imag = keras.layers.Multiply()( [amp, K.sin(phase)] )
#     # return keras.models.Model(inputs=mel, outputs=[spec_real, spec_imag])
#
#     log_amp_valid = keras.layers.Cropping1D((steps_leadin, 0), name='crop_a')(log_amp)
#     phase_shift_valid = keras.layers.Cropping1D((steps_leadin, 0), name='crop_p')(phase_shift)
#
#     # Concat the amps and phases into one return value
#     spec_concat = keras.layers.Concatenate(axis=-1, name='spec_concat')(
#         [log_amp_valid, phase_shift_valid])
#     return keras.models.Model(inputs=[mel_log, phase0], outputs=spec_concat)