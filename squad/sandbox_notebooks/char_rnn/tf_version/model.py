import tensorflow as tf
from tensorflow.contrib import rnn

class Model():
    def __init__(self, config, training=True):
        self.args = config
        if not training:
            self.args.batch_size = 1
            self.args.seq_length = 1

        # model_type
        if self.args.model_type.lower() == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.args.model_type.lower() == 'gru':
            cell_fn = rnn.GRUCell
        elif self.args.model_type.lower() == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif self.args.model_type.lower() == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type is not supported: {}".format(self.args.model_type.lower() ))

        # activation_type
        if self.args.activation.lower() == 'relu':
            activation = tf.nn.relu
        elif self.args.activation.lower() == 'tanh':
            activation = tf.nn.tanh
        else:
            raise Exception("Activation type is not supported: {}".format(self.args.activation.lower()))

        #optimizer
        if self.args.optimizer.lower() == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer
        elif self.args.optimizer.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer
        elif self.args.optimizer.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer
        elif self.args.optimizer.lower() == 'adagrad':
            optimizer = tf.train.AdagradOptimizer
        else:
            raise Exception("Optimizer is not supported: {}".format(self.args.optimizer.lower()))


        def build_cell(cell_fn, hidden_size, activation_fn, input_keep_prob, output_keep_prob,training_state):
            cell = cell_fn(num_units=hidden_size, activation=activation_fn)
            if training_state and (input_keep_prob < 1.0 or output_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell = cell,
                                          input_keep_prob=input_keep_prob,
                                          output_keep_prob=output_keep_prob)
            return cell

        with tf.name_scope('cell_fn'):
            # stack the cells and make them multirnncells
            self.cell = cell = rnn.MultiRNNCell([build_cell(cell_fn,
                                                            self.args.hidden_size,
                                                            activation,
                                                            self.args.keep_prop_output_layer,
                                                            self.args.keep_prop_input_layer,
                                                            training)
                                                 for _ in range(self.args.num_layers)
                                                     ])

            self.initial_cell_state = cell.zero_state(self.args.batch_size, tf.float32)

        with tf.name_scope('inputs'):
            # input tensor for storing input data
            self.input_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_length])
            # target tensor for storing target data
            self.target_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_length])



            if self.args.is_one_hot_embedding:
                embedded_inputs = tf.one_hot(name='embedding', indices=self.input_data, depth=self.args.vocab_size)
            else:
                # embedding tensor for converting input tensor to embedded input tensor (vocab_of_chars x for each cell hidden size)
                # hidden size can also interpreted as embedding size
                embedding = tf.get_variable("embedding", [self.args.vocab_size, self.args.hidden_size])
                embedded_inputs = tf.nn.embedding_lookup(embedding, self.input_data)


            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell,
                                                    inputs=embedded_inputs,
                                                    initial_state=self.initial_cell_state,
                                                    scope='rnnlm'
                                                    )
            #tf concat : t1 = [[1, 2, 3], [4, 5, 6]]
            #            t2 = [[7, 8, 9], [10, 11, 12]]
            # with axis = 1  # Result :  [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
            output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.args.hidden_size])
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(
                    initial_value=tf.truncated_normal(shape=[self.args.hidden_size, self.args.vocab_size], stddev=1.0))
                softmax_b = tf.Variable(tf.zeros(self.args.vocab_size))

            self.logits = tf.matmul(output, softmax_w) + softmax_b
            #self.logits_v2 = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

            self.probs = tf.nn.softmax(self.logits, name='predictions')

        with tf.name_scope('loss'):
            target_data_one_hot = tf.one_hot(self.target_data, self.args.vocab_size)
            target_data_one_hot_reshaped =  tf.reshape(target_data_one_hot, shape=self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=target_data_one_hot_reshaped)

            self.loss = tf.reduce_mean(loss)


        self.lr = tf.Variable(0.0, trainable=False)
        self.global_step = tf.Variable(0.0, trainable=False, name='global_step')
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), self.args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = optimizer(learning_rate=self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step)

        # #INSTRUMENT TENSORBOARD
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('loss', loss)
        # tf.summary.scalar('train_loss', self.cost)

    # def sample(self, sess, chars, vocab, num=200, prime='The', sampling_type=1):
    #     state = sess.run(self.cell.zero_state(1, tf.float32))
    #     for char in prime[:-1]:
    #         x = np.zeros((1,1))
    #         x[0,0] = vocab[char]
    #         feed = {self.input_data:x,
    #                 self.initial_cell_state: state}
    #         [state] = sess.run([self.final_state], feed)
    #
    #     def weighted_pick(weights):
    #         #cumulative sum
    #         t = np.cumsum(weights)
    #         s = np.sum(weights)
    #         return (int(np.searchsorted(t, np.random.rand(1) * s)))
    #
    #     ret = prime
    #     char = prime[-1]
    #
    #     for n in range(num):
    #         x = np.zeros((1,1))
    #         x[0,0] = vocab[char]
    #         feed = {self.input_data:x,
    #                 self.initial_cell_state:state}
    #         [probs, state] = sess.run([self.probs, self.final_state], feed)
    #         p = probs[0]
    #
    #         if sampling_type == 0:
    #             sample = np.argmax(p)
    #         elif sampling_type == 2:
    #             if char == ' ':
    #                 sample = weighted_pick(p)
    #             else:
    #                 sample = np.argmax(p)
    #         else:
    #             sample = weighted_pick(p)
    #
    #         pred = chars[sample]
    #         ret += pred
    #         char = pred
    #
    #     return ret