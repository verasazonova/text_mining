import logging
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from text_mining.corpora import medical
from text_mining.utils import ioutils as io
from text_mining.corpora.csv_tweet_reader import KenyanCSVMessage, IMDB
from text_mining.utils import textutils as tu
import argparse

__author__ = 'verasazonova'


def read_and_split_data(filename, p_labeled=1.0, p_used=0.0, n_trial=0, unlabeled_filenames=None, file_type=None):
    x_full, y_full, stoplist, ids = make_x_y(filename, ["text", "label"], file_type=file_type)

    print x_full.shape

    if unlabeled_filenames is not None and unlabeled_filenames:
        x_unlabeled = []
        for unlabeled in unlabeled_filenames:
            print "Unlabeled filenames:  ", unlabeled, file_type
            x, _, _, _ = make_x_y(unlabeled, ["text"], file_type=file_type)
            x_unlabeled += x
    else:
        x_unlabeled = []

    logging.info("Classifing for p= %s" % p_labeled)
    logging.info("Classifing for ntrials = %s" % n_trial)
    logging.info("Classifing for threshs = %s" % p_used)

    if p_labeled == 1:
        x_labeled = x_full
        y_labeled = y_full
        ids_l = ids
    else:
        x_unlabeled, x_labeled, y_unlabeled, y_labeled, _, ids_l = train_test_split(x_full, y_full, ids,
                                                                                    test_size=p_labeled,
                                                                                    random_state=n_trial)

    if p_used == 1:
        x_unlabeled_for_w2v = x_unlabeled
    else:
        x_unused, x_unlabeled_for_w2v = train_test_split(x_unlabeled, test_size=p_used, random_state=0)

    return x_labeled, y_labeled, x_unlabeled_for_w2v, ids_l


# read data from different formats
# extract the text (x), the label (y) and the id (optional)
def make_x_y(filename, fields=None, file_type=""):
#    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    if file_type=="tweets":
        stop_path = os.path.join(os.path.dirname(io.__file__), "en_swahili.txt")

        dataset = KenyanCSVMessage(filename, fields=fields, stop_path=stop_path)

        text_corpus = [tu.normalize_punctuation(tweet[dataset.text_pos]) for tweet in dataset]
        if dataset.label_pos is not None:
            labels = [tweet[dataset.label_pos] for tweet in dataset]
            classes, indices = np.unique(labels, return_inverse=True)
            # a hack to change the order
            #indices = -1*(indices - 1)
            print classes
            print np.bincount(indices)
        else:
            indices = None

        if dataset.id_pos is not None:
            ids = [tweet[dataset.id_pos] for tweet in dataset]
        else:
            ids = None
        stoplist =  dataset.stoplist

    elif file_type == "text":
        text_corpus = [tu.normalize_punctuation(text) for text in medical.PMCOpenSubset(filename)]
        indices=None
        stoplist=None
        ids=None

    elif file_type == "medab":
        stop_path = os.path.join(os.path.dirname(io.__file__), "stopwords_punct.txt")
        dataset = medical.MedicalReviewAbstracts(filename, ['T', 'A'], labeled=False, tokenize=False, stop_path=stop_path)
        text_corpus = [tu.normalize_punctuation(text) for text in dataset]
        labels = dataset.get_target()
        classes, indices = np.unique(labels, return_inverse=True)
        print classes
        print np.bincount(indices)
        ids = None
        stoplist = dataset.stoplist

    elif file_type == "imdb":
        stop_path = os.path.join(os.path.dirname(io.__file__), "stopwords_punct.txt")
        dataset = IMDB(filename)
        text_corpus = [tu.normalize_punctuation(text) for text in dataset.x]
        labels = dataset.y
        classes, indices = np.unique(labels, return_inverse=True)
        print classes
        print np.bincount(indices)
        ids = None
        stoplist = None

    else:
        ids = None
        stoplist = None
        text_corpus = None
        indices = None

    if ids is None:
        ids = np.zeros(len(indices))

    return np.array(text_corpus), indices, stoplist, ids


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--test', action='store', dest='test_filename', default="", help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('--p_labeled', action='store', dest='p', default='1', help='Fraction of labeled data')
    parser.add_argument('--p_used', action='store', dest='thresh', default='0', help='Fraction of unlabelled data')
    parser.add_argument('--ntrial', action='store', dest='ntrial', default='0', help='Number of the trial')
    parser.add_argument('--type', action='store', dest='filetype', default='tweets', help='Type of file: imdb, medab, text, tweets')

    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename="log.txt")

    # parameters for large datasets
    ntrial = int(arguments.ntrial)
    p_used = float(arguments.thresh)
    p_labelled = float(arguments.p)

    naming_dict = io.get_w2v_naming()

    x_labeled, y_labeled, x_unlabeled, _ = read_and_split_data(arguments.filename[0],
                                                                   p_labeled=p_labelled,
                                                                   p_used=p_used,
                                                                   n_trial=ntrial,
                                                                   unlabeled_filenames=arguments.filename[1:],
                                                                   file_type=arguments.filetype)

    # split, extract and normalize text training data
    io.save_data(x_labeled, naming_dict["x_train"])
    io.save_data(y_labeled, naming_dict["y_train"])

    x_test = []
    # extracts testing data if any
    if arguments.test_filename != "":
        test_filename = arguments.test_filename
        x_test, y_test, _,_ = read_and_split_data(test_filename,
                                                                   p_labeled=1,
                                                                   p_used=1,
                                                                   n_trial=ntrial,
                                                                   file_type=arguments.filetype)
        io.save_data(x_test, naming_dict["x_test"])
        io.save_data(y_test, naming_dict["y_test"])

    io.save_data(np.concatenate([x_labeled,x_unlabeled,x_test]), naming_dict["w2v_corpus"])


if __name__ == "__main__":
    __main__()