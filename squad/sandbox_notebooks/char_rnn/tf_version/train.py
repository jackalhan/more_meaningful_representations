from __future__ import print_function
import time
import os
import tensorflow as tf
from six.moves import cPickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from squad.sandbox_notebooks.char_rnn.tf_version.utils import TextLoader
from squad.sandbox_notebooks.char_rnn.tf_version.model import Model
import squad.sandbox_notebooks.char_rnn.tf_version.config as my_config


def main(_):
    config = my_config.flags.FLAGS
    print(vars(config)['__flags'])
    data_loader = TextLoader(config.data_dir,
                             config.data_file,
                             config.batch_size,
                             config.seq_length,
                             config.is_forced)
    config.vocab_size = data_loader.vocab_size

    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    with open(os.path.join(config.save_dir, "{}_config.pkl".format(data_loader.data_file_pre)), 'wb') as f:
        cPickle.dump(config, f)

    with open(os.path.join(config.save_dir,
                                           "{}_chars_vocab.pkl".format(data_loader.data_file_pre)), 'wb') as f:
        cPickle.dump((data_loader.chars,data_loader.vocab), f)
    with tf.Session() as sess:
        model = Model(config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # RESTORE THE MODEL
        if config.init_from is not None:
            ckpt = tf.train.get_checkpoint_state(config.init_from)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(config.num_epochs):
            model_lr = tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e))
            sess.run([model_lr])
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_cell_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data:x,
                        model.target_data:y,
                        model.initial_cell_state:state}

                #instribument for tensorboard
                _, step, new_state, loss = sess.run([
                    model.train_op, model.global_step, model.final_state, model.loss
                ], feed)
                end = time.time()
                current_step = tf.train.global_step(sess, model.global_step)

                if step % config.log_every == 0:
                    print('step: {}/{}... '.format(int(step), config.num_epochs * data_loader.num_batches),
                          'loss: {:.4f}... '.format(loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if current_step % config.save_every == 0:
                    checkpoint_path = os.path.join(config.save_dir, 'model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=current_step)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    tf.app.run()