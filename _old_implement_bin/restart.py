

import argparse
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.vocab_file, max_word_length)

    prefix = args.train_prefix

    kwargs = {
        'test': False,
        'shuffle_on_load': True,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(prefix, vocab, **kwargs)
    else:
        data = LMDataset(prefix, vocab, **kwargs)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    # set optional inputs
    if args.n_train_tokens > 0:
        options['n_train_tokens'] = args.n_train_tokens
    if args.n_epochs > 0:
        options['n_epochs'] = args.n_epochs
    if args.batch_size > 0:
        options['batch_size'] = args.batch_size

    train(options, data, args.n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/output/restart_checkpoint', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/output/restart_checkpoint/vocab-2016-09-10.txt', help='Vocabulary file')
    parser.add_argument('--train_prefix', default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/train_paragraphs/*', help='Prefix for train files')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_train_tokens', type=int, default=1686591)
    parser.add_argument('--n_epochs', type=int, default=1)

    args = parser.parse_args()
    main(args)

