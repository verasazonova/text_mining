import logging
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from text_mining.corpora import medical
from text_mining.utils import ioutils

__author__ = 'verasazonova'


def read_and_split_data(filename, p=1, thresh=0, n_trial=0, unlabeled_filenames=None, dataname=""):
    x_full, y_full, stoplist, ids = make_x_y(filename, ["text", "label"])

    if unlabeled_filenames is not None:
        x_unlabeled = []
        for unlabeled in unlabeled_filenames:
            if not os.path.basename(unlabeled).startswith("units_"):
                file_type = "text"
            else:
                file_type = "medical"
            print "Unlabeled filenames:  ", unlabeled, file_type
            x, _, _, _ = make_x_y(unlabeled, ["text"], file_type=file_type)
            x_unlabeled += x
    else:
        x_unlabeled = []

    logging.info("Classifing for p= %s" % p)
    logging.info("Classifing for ntrials = %s" % n_trial)
    logging.info("Classifing for threshs = %s" % thresh)

    if p == 1:
        x_labeled = x_full
        y_labeled = y_full
        ids_l = ids
    else:
        x_unlabeled, x_labeled, y_unlabeled, y_labeled, _, ids_l = train_test_split(x_full, y_full, ids, test_size=p,
                                                                                    random_state=n_trial)

    if thresh == 1:
        x_unlabeled_for_w2v = x_unlabeled
    else:
        x_unused, x_unlabeled_for_w2v = train_test_split(x_unlabeled, test_size=thresh, random_state=0)

    experiment_name = "%s_%0.3f_%0.1f_%i" % (dataname, p, thresh, n_trial)

    return x_labeled, y_labeled, x_unlabeled_for_w2v, experiment_name, stoplist, ids_l


def make_x_y(filename, fields=None, file_type="medical"):
#    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    if file_type=="tweets":
        stop_path = os.path.join(os.path.dirname(ioutils.__file__), "en_swahili.txt")

        dataset = ioutils.KenyanCSVMessage(filename, fields=fields, stop_path=stop_path)

        tweet_text_corpus = [tweet[dataset.text_pos] for tweet in dataset]
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
        tweet_text_corpus = [text for text in medical.PMCOpenSubset(filename)]
        indices=None
        stoplist=None
        ids=None

    else:
        stop_path = os.path.join(os.path.dirname(ioutils.__file__), "stopwords_punct.txt")
        dataset = medical.MedicalReviewAbstracts(filename, ['T', 'A'], labeled=False, tokenize=False, stop_path=stop_path)
        tweet_text_corpus = [text for text in dataset]
        labels = dataset.get_target()
        classes, indices = np.unique(labels, return_inverse=True)
        print classes
        print np.bincount(indices)
        ids = None
        stoplist = dataset.stoplist

    return tweet_text_corpus, indices, stoplist, ids