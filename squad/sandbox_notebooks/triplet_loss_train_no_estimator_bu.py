"""Train the model"""

import argparse
import os
import math
import tensorflow as tf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from triplet_loss.utils import Params, train_test_splitter, dump_embeddings, analyze_labes
from triplet_loss.triplet_loss import batch_all_triplet_loss
from triplet_loss.triplet_loss import batch_hard_triplet_loss, lossless_triplet_loss
from triplet_loss.quadratic_loss import euclidean_distance_loss
from triplet_loss.utils import dump_embeddings
import numpy as np
import triplet_loss.my_dataset as ds

parser = argparse.ArgumentParser()

# model path, the pretrained embeddings for questions, paragraphs and their mappings label file
parser.add_argument('--model_dir',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function',
                    help="Experiment directory containing params.json")
parser.add_argument('--question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
                    help="qustion embeddings_file")
parser.add_argument('--paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_paragraph_embeddings.hdf5',
                    help="paragraph embeddings_file")
parser.add_argument('--labels_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_q_to_p_mappings.csv',
                    help="labels_file")

# run configrations
parser.add_argument('--is_train',
                    default=False,
                    help="Run for the training")
parser.add_argument('--is_test',
                    default=False,
                    help="Run for the testing")
parser.add_argument('--is_prediction_for_evaluation',
                    default=False,
                    help="Run eval for the prediction")
parser.add_argument('--is_recall_comparision_with_baseline',
                    default=True,
                    help="Recall comparision with baseline")

parser.add_argument('--is_prediction_for_live',
                    default=True,
                    help="Run live for the prediction")
parser.add_argument('--is_dump_predictions',
                    default=True,
                    help="whether dump the prediction or not")

# data train/eval split configrations
parser.add_argument('--split_train_test',
                    default=False,
                    help="control whether split the dataset")
parser.add_argument('--analyze_labels',
                    default=False,
                    help="analyze the labels (input) so that we can balance the data")
parser.add_argument('--limit_data',
                    default=None,
                    help="Limit the data based on number of paragraph size for debug purposes. None or Int")
# parser.add_argument('--train_splitter_rate',
#                     default=0.6,
#                     help="how much of the data to be used as train")
# parser.add_argument('--eval_question_size_for_recall',
#                     default=2000,
#                     help="how much of the data to be used as train")

# if args.split_train_test is False, data is already splitted,
# file locations of the splitted data: Train Ques/Par Embeddings, Test Ques/Par Embeddings

# TEST/EVAL
parser.add_argument('--test_question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_question_embeddings.hdf5',
                    help="Test/Eval question embeddings data")
parser.add_argument('--test_paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_paragraph_embeddings.hdf5',
                    help="Test/Eval paragraph embeddings data")
parser.add_argument('--test_label_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_q_to_p_mappings.csv',
                    help="Test/Eval paragraph embeddings data")

# TEST/EVAL RECALL
parser.add_argument('--test_recall_question_embeddings',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_test_recall_question_embeddings.hdf5',
                    help="Test/Eval question embeddings data for recall")

parser.add_argument('--test_recall_paragraph_embeddings',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_paragraph_embeddings.hdf5',
                    help="Test/Eval paragraph embeddings data for recall")

# TRAIN
parser.add_argument('--train_question_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_question_embeddings.hdf5',
                    help="Train question embeddings data")
parser.add_argument('--train_paragraph_embeddings_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_paragraph_embeddings.hdf5',
                    help="Train paragraph embeddings data")
parser.add_argument('--train_label_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/splitted_train_q_to_p_mappings.csv',
                    help="Train paragraph embeddings data")

# DATA to be predicted (ALL QUESTIONS)
parser.add_argument('--pretrained_embedding_file',
                    default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
                    help="pretrained embeddings file")


def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    with tf.name_scope('l2_form') as scope:
        square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True, name='square_sum')
        norm = tf.sqrt(tf.maximum(square_sum, tf.keras.backend.epsilon()), name='norm')
    return norm


def pairwise_cosine_sim(A, B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions
    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point
    """
    with tf.name_scope('pairwise_cosine_sim') as scope:
        A_mag = l2_norm(A, axis=2)
        B_mag = l2_norm(B, axis=2)
        num = tf.keras.backend.batch_dot(A, tf.keras.backend.permute_dimensions(B, (0, 2, 1)))
        den = (A_mag * tf.keras.backend.permute_dimensions(B_mag, (0, 2, 1)))
        dist_mat = num / den
    return dist_mat


def calculate_recalls(questions, paragraphs, labels, params, k=None, extract_type=1):
    """
     question [n x d] tensor of n rows with d dimensions
     paragraphs [m x d] tensor of n rows with d dimensions
     params config
     returns:
     loss : scalar value
     """

    with tf.name_scope('recall_loss') as scope:
        recalls = []  # tf.zeros([len(params.recall_top_k), 1], tf.float32)

        # in order to support batch_size feature, we expanded dims for 1
        paragraphs = tf.expand_dims(paragraphs, axis=0)
        labels = tf.expand_dims(labels, axis=0)
        questions = tf.expand_dims(questions, axis=0)
        number_of_questions = tf.to_int64(tf.shape(questions)[1])
        # now question, paragraphs pairwise calculation
        distances = pairwise_cosine_sim(questions, paragraphs)
        for _k in [k] if k is not None else params.recall_top_k:
            with tf.name_scope('top_k_{}'.format(_k)) as k_scope:
                # find the top_k paragraphs for each question
                top_k = tf.nn.top_k(distances, k=_k, name='top_k_top_k_{}'.format(_k))

                # is groundtruth label is in these top_k paragraph
                equals = tf.equal(top_k.indices, labels, name='equal_top_k_{}'.format(_k))

                # cast the equals to int32 to count the non zero ones because if it is equal,
                # there is only one 1 for each question among paragraphs,
                # then label is in top k
                casted_equal = tf.cast(equals, dtype=tf.int32, name='casted_equal_top_k_{}'.format(_k))
                final_equals_non_zero = tf.squeeze(
                    tf.count_nonzero(casted_equal, axis=2, name='sq_top_k_{}'.format(_k)))

                # get the details of true question - paragraph
                indx_of_questions_that_has_the_correct_paragraphs = tf.reshape(
                    tf.squeeze(tf.where(tf.equal(final_equals_non_zero, extract_type))), shape=[-1, 1])
                top_k_values = tf.reshape(tf.squeeze(top_k.values), shape=[-1, 1])
                cos_values_of_that_has_the_correct_paragraphs = tf.reshape(tf.to_float(tf.gather(top_k_values,
                                                                                                 indx_of_questions_that_has_the_correct_paragraphs)),
                                                                           shape=[-1, 1])
                label_values = tf.reshape(tf.squeeze(labels), shape=[-1, 1])
                label_values_of_that_has_the_correct_paragraphs = tf.reshape(tf.to_float(tf.gather(label_values,
                                                                                                   indx_of_questions_that_has_the_correct_paragraphs)),
                                                                             shape=[-1, 1])
                question_index_labels_and_scores_that_has_the_correct_paragraphs = tf.concat(
                    [tf.reshape(tf.to_float(indx_of_questions_that_has_the_correct_paragraphs),shape=[-1,1]),
                     label_values_of_that_has_the_correct_paragraphs,
                     cos_values_of_that_has_the_correct_paragraphs
                     ], axis=1)



                total_founds_in_k = tf.reduce_sum(final_equals_non_zero)
                recalls.append(total_founds_in_k)

        recalls = tf.stack(recalls)
        best_possible_score = len(params.recall_top_k) * number_of_questions
        current_score = tf.reduce_sum(recalls)
        loss = (current_score / best_possible_score)

    return loss, recalls, (
                recalls / number_of_questions), number_of_questions, question_index_labels_and_scores_that_has_the_correct_paragraphs


def define_closest_paragraphs(batch, paragraphs):

    sub_set_ = batch
    sub_set_ = tf.expand_dims(sub_set_, axis=0)
    actual_set = tf.expand_dims(paragraphs, axis=0)

    dist = pairwise_cosine_sim(sub_set_, actual_set)
    top_k = tf.nn.top_k(dist, k=2, name='top_k_{}'.format(1))
    values = tf.reshape(tf.reduce_mean(top_k.values, axis=2),shape=[tf.shape(sub_set_)[1], 1])
    indices = tf.to_float(tf.reshape(tf.squeeze(top_k.indices[:,:,1]), shape=[-1,1]))
    values = tf.reshape(tf.squeeze(values), shape=[-1,1])
    top_k = tf.concat((values, indices), axis=1)
    return top_k

def define_closest_paragraphs_to_questions(batch, paragraphs):

    sub_set_ = batch
    sub_set_ = tf.expand_dims(sub_set_, axis=0)
    actual_set = tf.expand_dims(paragraphs, axis=0)

    dist = pairwise_cosine_sim(sub_set_, actual_set)
    top_k = tf.nn.top_k(dist, k=1, name='top_k_{}'.format(1))
    values = tf.reshape(tf.reduce_mean(top_k.values, axis=2),shape=[tf.shape(sub_set_)[1], 1])
    indices = tf.to_float(tf.reshape(tf.squeeze(top_k.indices), shape=[-1,1]))
    values = tf.reshape(tf.squeeze(values), shape=[-1,1])
    top_k = tf.concat((values, indices), axis=1)
    return top_k


def next_batch(begin_indx, batch_size, questions, labels, paragraphs):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(begin_indx, begin_indx+batch_size)
    np.random.shuffle(idx)
    questions = questions[idx]
    labels = labels[idx]
    paragraphs = paragraphs[idx]

    return questions, labels, paragraphs

if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    if args.analyze_labels:
        analysis = analyze_labes(args.labels_file)

    else:
        if args.split_train_test:
            file_paths = train_test_splitter(args.question_embeddings_file,
                                             args.paragraph_embeddings_file,
                                             args.labels_file,
                                             params.train_splitter_rate,
                                             params.eval_question_size_for_recall,
                                             args.limit_data)
            params.train_size = file_paths['train_question_size']
            params.eval_size = file_paths['eval_question_size']
            params.num_labels = file_paths['num_labels']
            params.save(json_path)
        else:
            file_paths = {}
            file_paths['train_question_embeddings'] = args.train_question_embeddings_file
            file_paths['train_paragraph_embeddings'] = args.train_paragraph_embeddings_file
            file_paths['train_paragraph_labels'] = args.train_label_file

            file_paths['test_question_embeddings'] = args.test_question_embeddings_file
            file_paths['test_paragraph_embeddings'] = args.test_paragraph_embeddings_file
            file_paths['test_paragraph_labels'] = args.test_label_file

            file_paths['test_recall_question_embeddings'] = args.test_recall_question_embeddings
            file_paths['paragraph_embeddings'] = args.paragraph_embeddings_file
            # Define the model
            tf.logging.info("Creating the model...")

            questions = tf.placeholder(tf.float32, [None, params.embedding_dim])
            labels = tf.placeholder(tf.int32, [None, 1])
            paragraphs = tf.placeholder(tf.float32, [None, params.embedding_dim])
            normalized_paragraphs = tf.nn.l2_normalize(paragraphs, name='pars', axis=1)
            # after_normalized_paras = tf.sqrt(tf.reduce_sum(tf.square(normalized_paragraphs), axis=1))
            # bnp = anp = tf.Print(after_normalized_paras, [after_normalized_paras], 'after_normalized_paras')
            # with tf.variable_scope("queue"):
            #     q = tf.FIFOQueue(capacity=5, dtypes=tf.float32)  # enqueue 5 batches
            #     # We use the "enqueue" operation so 1 element of the queue is the full batch
            #     enqueue_op_q = q.enqueue(questions_input)
            #     enqueue_op_l = q.enqueue(labels_input)
            #     enqueue_op_p = q.enqueue(paragraphs_input)
            #     numberOfThreads = params.num_parallel_calls
            #     qr = tf.train.QueueRunner(q, [enqueue_op_q, enqueue_op_l, enqueue_op_p] * numberOfThreads)
            #     tf.train.add_queue_runner(qr)
            #     data = q.dequeue()  # It replaces our input placeholder
            #     questions = data[0]
            #     labels = data[1]
            #     paragraphs = data[2]
            with tf.variable_scope('model'):
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

            embedding_mean_norm = tf.reduce_mean(tf.norm(out, axis=1))
            tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
            with tf.variable_scope('Loss'):
                    # Define triplet loss
                if params.triplet_strategy == "batch_all":
                    loss, fraction = batch_all_triplet_loss(out, normalized_paragraphs, tf.reshape(labels, shape=[-1]), margin=params.margin,
                                                            squared=params.squared)
                elif params.triplet_strategy == "batch_hard":
                    loss = batch_hard_triplet_loss(out, normalized_paragraphs, tf.reshape(labels, shape=[-1]), weights=fc1, params=params,
                                                   squared=params.squared)
                elif params.triplet_strategy == 'semi_hard':
                    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.reshape(labels, shape=[-1]), out)
                elif params.triplet_strategy == 'lossless_triplet':
                    loss = lossless_triplet_loss(out, normalized_paragraphs, labels, params, squared=params.squared, margin=params.margin)
                elif params.triplet_strategy == 'quadratic_reg_loss':
                    loss = euclidean_distance_loss(out, normalized_paragraphs, fc1, params)
                elif params.triplet_strategy == 'abs_reg_loss':
                    loss = euclidean_distance_loss(out, normalized_paragraphs, fc1, params,type='abs')
                else:
                    raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))
                # add loss to collections
                tf.losses.add_loss(loss)
                    # # -----------------------------------------------------------
                    # # METRICS AND SUMMARIES
                    # # Metrics for evaluation using tf.metrics (average over whole dataset)
                    # # TODO: some other metrics like rank-1 accuracy?
            with tf.variable_scope("metrics"):
                eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}
                if params.triplet_strategy == "batch_all":
                    eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

            with tf.variable_scope('Accuracy'):
                eval_loss, recalls, normalized_recalls, number_of_questions, q_index_and_cos = calculate_recalls(out, normalized_paragraphs, labels, params)
                accuracy = eval_loss
                accuracy = tf.Print(accuracy, data=[accuracy], message="Average Recall:")

                # Summaries for training
            tf.summary.scalar('loss', loss)
            if params.triplet_strategy == "batch_all":
                tf.summary.scalar('fraction_positive_triplets', fraction)
            if params.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
            global_step = tf.train.get_global_step()
            train_op = optimizer.minimize(tf.losses.get_total_loss(), global_step=global_step)
            # finally setup the initialisation operator
            init_op = tf.global_variables_initializer()

            # start the session
            with tf.Session() as sess:
                sess.run(init_op)
                # ... check the accuracy before training, ...

                q = ds._load_embeddings(file_paths['train_question_embeddings'])
                l = q[:, params.embedding_dim:params.embedding_dim+1]
                q = q[:, :params.embedding_dim]
                p = ds._load_embeddings(file_paths['train_paragraph_embeddings'])

                q_, l_, _ = next_batch(10, params.eval_question_size_for_recall, q, l, p)
                recall_p = ds._load_embeddings(args.paragraph_embeddings_file) #accuracy
                # # ... add the coordinator, ...
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(coord=coord)

                # acc, rec, norm_rec = sess.run([accuracy,recalls, normalized_recalls], feed_dict={
                #     questions: q_,
                #     paragraphs: recall_p,
                #     labels: l_
                # })
                # print(rec)
                # print(norm_rec)
                test_q = ds.get_question_embeddings(False, file_paths['test_question_embeddings'], params)
                test_l = test_q[:, params.embedding_dim:params.embedding_dim + 1]
                test_q = test_q[:, :params.embedding_dim]


                total_batch = int(len(l) / params.batch_size)
                for epoch in range(params.num_epochs):
                    print("Epoch:", (epoch + 1), "is started")
                    avg_cost = 0
                    for i in range(total_batch):
                        #  ... without sampling from Python and without a feed_dict !
                        q_sub, l_sub, p_sub = next_batch(i*params.batch_size, params.batch_size, q, l, p)

                        _, my_loss, new_question_embeddings = sess.run([train_op, loss, out], feed_dict={
                            questions: q_sub,
                            paragraphs: p_sub,
                            labels: l_sub
                        })

                        variables_names = [v.name for v in tf.trainable_variables()]
                        values = sess.run(variables_names)

                        for k, v in zip(variables_names, values):
                            if k == 'model/model/fc/_fc1/weights:0':
                                x = np.sum(np.mean(v,axis=1))

                        # We regularly check the loss
                        if i % 100 == 0:
                            print('iter:%d - loss:%f' % (i, my_loss))

                        avg_cost += my_loss / total_batch

                    print('Model Performence on the set of {}'.format(params.eval_question_size_for_recall))
                    sess.run(accuracy, feed_dict={
                        questions: test_q,
                        paragraphs: recall_p,
                        labels: test_l
                    })
                    rec, norm_rec = sess.run([recalls, normalized_recalls, ], feed_dict={
                        questions: test_q,
                        paragraphs: recall_p,
                        labels: test_l
                    })
                    print(rec)
                    print(norm_rec)

                    print("Epoch:", (epoch + 1), "avg loss =", "{:.3f}".format(avg_cost))
                    print("*" * 25)
                    #
                    # sess.run(accuracy, feed_dict={
                    #     questions: q_,
                    #     paragraphs: recall_p,
                    #     labels: l_
                    # })
                raw_ques = ds._load_embeddings(args.pretrained_embedding_file)
                new_question_embeddings = sess.run(out, feed_dict={
                    questions: raw_ques
                })

                print('New Embeddings are predicted')
                print('{}'.format(raw_ques.shape))

                _e = args.pretrained_embedding_file.rpartition(os.path.sep)
                path_e = _e[0]
                new_embedding_embed_file = os.path.join(path_e, 'improved_' + _e[2].replace('train', ''))
                dump_embeddings(new_question_embeddings, new_embedding_embed_file)

    print('Done')

