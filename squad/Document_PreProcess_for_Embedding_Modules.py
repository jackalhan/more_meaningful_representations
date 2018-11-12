import math
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import helper.utils as UTIL
import argparse

TRAIN = 'train'
DEV = 'dev'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_length', default=3, type=int, help="-1: no windows, else should be selected from the given range 1-512. it should be at least 1 lower than truncate_length")
    parser.add_argument('--truncate_length', default=512, type=int, help="-1: unlimited, else length should be selected from the given range 1-512.")
    parser.add_argument('--data_path', help="path to the source file to be converted")
    parser.add_argument('--from_file_name', help="squad formatted dataset file name with json extension")
    return parser

def main(args):
    ################ CONFIGURATIONS #################
    source_file = os.path.join(args.data_path, args.from_file_name)
    source_root_file_name = args.from_file_name.rpartition(os.path.sep)[-1].rpartition('.')[:-2][0]
    ################ ALGOS #################

    """
    ******************************************************************************************************************
    START: PARSING FILE
    ******************************************************************************************************************
    """
    tokenized_questions, tokenized_paragraphs, questions_nontokenized, paragraphs_nontokenized = UTIL.prepare_squad_objects(source_file, source_root_file_name)

    """
    ******************************************************************************************************************
    END: PARSING FILE
    ******************************************************************************************************************
    """

    """
    ******************************************************************************************************************
    START: SLIDING WINDOW
    ******************************************************************************************************************
    """
    get_slideed_tokenizations_and_dump(tokenized_questions, UTIL.create_dir(os.path.join(args.data_path, 'questions_windowed')), args.truncate_length, args.window_length)

    get_slideed_tokenizations_and_dump(tokenized_paragraphs, UTIL.create_dir(os.path.join(args.data_path, 'paragraphs_windowed')), args.truncate_length,
                                                               args.window_length)


    """
    ******************************************************************************************************************
    END: SLIDING WINDOW
    ******************************************************************************************************************
    """

def get_slideed_tokenizations_and_dump(tokenized_contents, save_path, truncate_length, window_length):

    for indx, content in enumerate(tqdm(tokenized_contents), 1):

        iteration_size = math.ceil((len(content) - window_length) / (truncate_length - window_length))
        iteration_size = 1 if iteration_size == 0 else iteration_size
        indexer = np.arange(truncate_length)[None, :] + (truncate_length - window_length) * np.arange(iteration_size)[:,
                                                                                            None]
        additional_chars = np.amax(indexer) - len(content) + 1
        new_content = content + [None for _ in range(additional_chars)]
        _content = np.array(new_content)
        #temp_new_tokenized = np.expand_dims(_content[indexer], axis=0)
        # if new_tokenized is None:
        #     new_tokenized = temp_new_tokenized
        # else:
        #     new_tokenized = np.vstack((new_tokenized, temp_new_tokenized))
        processed_content = _content[indexer]
        UTIL.dump_tokenized_contexts(processed_content, os.path.join(save_path, str(indx) + '.txt'), True)

if __name__ == '__main__':
    """
    sample executions: 
    
    """
    args = get_parser().parse_args()
    assert args.data_path is not None, "No folder path found at {}".format(args.data_path)
    assert args.from_file_name is not None, "No 'from_file_name' found {}".format(args.from_file_name)
    #assert args.to_file_name is not None, "No 'to_file_name' found {}".format(args.to_file_name)
    main(args)
