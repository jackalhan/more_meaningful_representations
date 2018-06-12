import argparse

def get_parser():
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

    parser.add_argument('--is_prediction',
                        default=False,
                        help="Run live for the prediction")

    parser.add_argument('--is_run_metrics',
                        default=True,
                        help="Run live for the prediction")

    # parser.add_argument('--is_dump_predictions',
    #                     default=True,
    #                     help="whether dump the prediction or not")

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
    # parser.add_argument('--pretrained_embedding_file',
    #                     default='/home/jackalhan/Development/github/more_meaningful_representations/squad/train/triplet_loss_function/data/train_question_embeddings.hdf5',
    #                     help="pretrained embeddings file")

    return parser