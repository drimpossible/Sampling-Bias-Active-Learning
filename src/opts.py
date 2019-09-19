import argparse


def myargparser():
    parser = argparse.ArgumentParser(description='Active Learning')

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--dataset', type=str, default='ag_news', choices=['trec_qa', 'sogou_news', 'dbpedia',
                                                                           'yahoo_answers', 'yelp_review_polarity',
                                                                           'yelp_review_full', 'ag_news',
                                                                           'amazon_review_polarity',
                                                                          'amazon_review_full'], help='Name of dataset Options: agnews')
    data_args.add_argument('--data_dir', type=str, default='../data/', help='Data directory (Eg: ../data/)')
    data_args.add_argument('--num_classes', type=int, default=-1, help='Number of classes in the dataset')

    data_args.add_argument('--num_points', type=int, default=0, help='Number of train samples')
    data_args.add_argument('--num_test', type=int, default=0, help='Number of testing samples')

    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--model', type=str, default='FastText', choices=['FastText', 'NaiveBayes', 'LinearSVM', 'BERTL'],
                            help='Name of model Options: "FastText", "NaiveBayes"')
    model_args.add_argument('--dim', type=int,  default=25, help='Number of testing samples')
    model_args.add_argument('--num_buckets', type=int, default=10000000, help='Number of testing samples')
    model_args.add_argument('--min_count', type=int, default=1, help='Number of testing samples')
    model_args.add_argument('--num_ngrams', type=int, default=2, help='Number of testing samples')
    model_args.add_argument('--pre_trained_path', type=str, default=None, help='Use 300d pre-trained vectors')
    model_args.add_argument('--quantize', type=bool, default=True, help='Quantize model')
    model_args.add_argument('--qnorm', type=bool, default=True, help='Norm quantization')
    model_args.add_argument('--retrain_quantize', type=bool, default=True, help='Retrain after quantization')
    model_args.add_argument('--cutoff', type=int, default=100000, help='Retrain after quantization')
    model_args.add_argument('--qout', type=bool, default=False, help='Quantize Classifier')
    model_args.add_argument('--num_ensemble', type=int, default=1, help='Number of classifiers')

    misc_args = parser.add_argument_group('Cleaning and backup arguments')
    misc_args.add_argument('--seed', type=int, default=0, help='Seed value for training')
    misc_args.add_argument('--logpath', type=str, default='../logs/', help='Logging directory (../logs)')
    misc_args.add_argument('--workers', type=int, default=8, help='Number of parallel worker threads')
    misc_args.add_argument('--exp_name', type=str, default='test_debug_ag_news', help='Name of experiment')

    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train the model')
    optim_args.add_argument('--lr', type=float, default=0.75, help='Number of epochs to train the model')

    replicate_args = parser.add_argument_group('Replication experiments')
    replicate_args.add_argument('--expmode', type=str, choices=['train','replicate'], default='train', help='Mode')
    replicate_args.add_argument('--expload', type=str, help='The path to the expname of experiment you want to replicate')

    activ_args = parser.add_argument_group('Active Learning related arguments')
    activ_args.add_argument('--init_train_percent', type=float, default=0.01, help='% of training data to start training')
    activ_args.add_argument('--query_iter', type=int, default=39, help='Number of acquisition iterations')
    activ_args.add_argument('--query_type', type=str, default='entropy', help='Acquisition Function')
    activ_args.add_argument('--num_acquise_percent', type=float, default=0.01, help='Number of acquisitions for each query')
    activ_args.add_argument('--best_label_entropy', type=float, default=0, help='Highest possible value of label entropy for the dataset')
    activ_args.add_argument('--best_accuracy', type=float, default=0, help='Accuracy of supervised fasttext model')
    activ_args.add_argument('--num_delete_percent',  type=float, default=None, help='Number of deletions for each query')
    misc_args.add_argument('--epoch_num', type=int, default=15, help='Name of experiment')
    return parser

