import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path',
                        default="/home/jackalhan/Development/github/more_meaningful_representations/squad/train/improvement/params.json",
                        help="path to the config file")

    return parser