from text_mining.experiments.build_model import build_w2v_model
from text_mining.experiments.run_classification import explore_classifier, run_cv_classifier, run_train_test_classifier
from text_mining.experiments.treat_data import read_and_split_data, make_x_y
from text_mining.experiments.vectorize_data import build_and_vectorize_w2v, scale_features, build_experiments
from text_mining.models import w2v_models, cluster_models, transformers, lda_models

__author__ = 'verasazonova'

import argparse
import sklearn.metrics
import logging
import os
import numpy as np
import re
from text_mining.utils import ioutils, plotutils
from text_mining.utils import textutils as tu
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time


def calculate_and_plot_lda(filename, ntopics, dataname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/en_swahili.txt"

    # Load dataset
    dataset = ioutils.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"], stop_path=stop_path)

    # Unless the counts and topic definitions have already been extracted
    if not os.path.isfile(dataname+"_cnts.txt"):
        # Create the histogram of LDA topics by date
        lda_models.bin_tweets_by_date_and_lda(dataset, n_topics=ntopics, mallet=False, dataname=dataname)

    # Read the resulting counts, date bins, and topics
    counts, bins, topics = ioutils.read_counts_bins_labels(dataname)

    # Figure out which topics to cluster together
    clustered_counts, clustered_labels, clusters = cluster_models.build_clusters(counts, topics, thresh=0.09)

    # Plot the clustered histogram
    plotutils.plot_tweets(counts=clustered_counts, dates=bins, clusters=clusters,
                          labels=clustered_labels, dataname=dataname)

    flattened_topic_list = [word for topic in topics for word, weight in topic]
    print len(flattened_topic_list)


def extract_phrases(tweet_text_corpus, stoplist):
    for thresh in [6000, 7000, 8000, 10000]:
        print "Threshhold %i " % thresh
        text_corpus, dictionary, bow = tu.process_text(tweet_text_corpus, stoplist=stoplist,
                                                       bigrams=thresh, trigrams=None, keep_all=False,
                                                       no_below=10, no_above=0.8)

        bigrams = [word for word in dictionary.token2id.keys() if re.search("_", word)]
        print len(bigrams)
        print ", ".join(bigrams)

        print



def tweet_bow_classification(filename, dataname, n_trial=None,  p=None, thresh=None,
                             clf_name='bow', clf_base="lr", action="classify"):
    x_data, y_data, unlabeled_data, run_dataname, stoplist, ids = read_and_split_data(filename=filename, p=p, thresh=thresh,
                                                                                  n_trial=n_trial, dataname=dataname)

    bow = transformers.BOWModel(no_above=0.8, no_below=8, stoplist=stoplist)

    # get matrices of features from x_data
    data = bow.fit_transform(x_data)

    if clf_base == "lr":
        clf = LogisticRegression()
    elif clf_base == "sdg":
        clf = sklearn.linear_model.SGDClassifier(loss='log', penalty="l2",alpha=0.005, n_iter=5, shuffle=True)
    else:
        clf = SVC(kernel='linear', C=10)

    name="BOW"
    if action == "classify":

        scores = run_cv_classifier(data, y_data, clf=clf, n_trials=10, n_cv=3, direct=False)
        print name, np.mean(scores, axis=0), scores.shape


        with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:

            for i, score in enumerate(scores):
                f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %f, %f, %f, %f, %i, %s \n" %
                       (n_trial, i, name, -1, p, thresh, data.shape[0], data.shape[0]*(p+thresh-p+thresh),
                        score[0], score[1], score[2], score[3], score[4], -1, clf_base))
            f.flush()


def tweet_classification(filename, size, window, dataname, p=None, thresh=None, n_trial=None, clf_name='w2v',
                         unlabeled_filenames=None, clf_base="lr", action="classify", rebuild=False, min_count=1,
                         recluster_thresh=0, n_components=30, experiment_nums=None, test_filename=None,
                         diff1_max=3, diff0_max=1):

    experiment_name = "%s_%0.3f_%0.1f_%i" % (dataname, p, thresh, n_trial)

    start_time = time.time()


    w2v_data_name = dataname+"_w2v_data"
    w2v_data_scaled_name = "%s_%i_%i_%i_scaled_w2v_data" % (experiment_name, size, window, min_count)
    y_data_name = "%s_%i_%i_%i_y_data" % (experiment_name, size, window, min_count)
    w2v_feature_crd_name = "%s_%i_%i_%i_w2v_f_crd" % (experiment_name, size, window, min_count)
    ids = []
    x_data = []

    if not os.path.isfile(w2v_data_scaled_name+".npy"):

        x_data, y_data, unlabeled_data, run_dataname, stoplist, ids = read_and_split_data(filename=filename, p=p, thresh=thresh,
                                                                                  n_trial=n_trial, dataname=dataname,
                                                                                  unlabeled_filenames=unlabeled_filenames)

        train_data_end = len(y_data)

        if test_filename is not None:
            x_test, y_test, _, _ = make_x_y(test_filename,["text", "label", "id_str"])
            x_data = np.concatenate([x_data, x_test])
            y_data = np.concatenate([y_data, y_test])
            print x_data.shape, y_data.shape


        #x_data, y_data, unlabeled_data, run_dataname, stoplist = read_and_split_data(filename=filename, p=p, thresh=thresh,
        #                                                                      n_trial=n_trial, dataname=dataname)

        # should make this into a separate process to release memory afterwards
        w2v_data, w2v_feature_crd = build_and_vectorize_w2v(x_data=x_data, y_data=y_data,
                                                              unlabeled_data=unlabeled_data, window=window,
                                                              size=size, dataname=run_dataname,
                                                              rebuild=rebuild,action=action,
                                                              stoplist=stoplist, min_count=min_count,
                                                              diff1_max=diff1_max, diff0_max=diff0_max)

        # scale
        print "Vectorized.  Saving"
        logging.info("Vectorized. Saving")
        np.save(w2v_data_name, np.ascontiguousarray(w2v_data))

        pickle.dump(w2v_feature_crd, open(w2v_feature_crd_name, 'wb'))

        logging.info("Scaling")
        print "Scaling"

        w2v_data = scale_features(w2v_data, w2v_feature_crd)
        #dpgmm_data = scale_features(dpgmm_data, dpgmm_feature_crd)
        print "Scaled. Saving"
        logging.info("Scaled. Saving")

        if os.path.isfile(w2v_data_name+".npy"):
            os.remove(w2v_data_name+".npy")
        np.save(w2v_data_scaled_name, np.ascontiguousarray(w2v_data))
        np.save(y_data_name, np.ascontiguousarray(y_data))

        print "Building experiments"
        logging.info("Building experiments")

    else:

        print("%s s: " % (time.time() - start_time))
        w2v_data = np.load(w2v_data_scaled_name+".npy", mmap_mode='c')
        # this need to be C-order array.
        y_data = np.load(y_data_name+".npy")

        print "loaded data %s" % w2v_data
        print("%s s: " % (time.time() - start_time))
        w2v_feature_crd = pickle.load(open(w2v_feature_crd_name, 'rb'))
        print "Loaded feature crd %s" % w2v_feature_crd
        train_data_end = int(p*1600000)
        logging.info("Loaded data, features.  %s " %  str(w2v_data.shape))



    names, experiments = build_experiments(w2v_feature_crd, experiment_nums=experiment_nums)
    print("%s s: " % (time.time() - start_time))

    print "Built experiments: ", names
    print experiments
    print action
    print train_data_end, w2v_data.shape

    logging.info("Built experiments: %s" % str(names))
    logging.info("Proceeding with %s %i %s"  % (action, train_data_end, str(w2v_data.shape)))

    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
        for name, experiment in zip(names, experiments):
            print("%s s: " % (time.time() - start_time))
            print name, experiment
            logging.info("Experiment %s %s" % (name, str(experiment)))
            #inds = []
            #for start, stop in experiment:
            #    inds += (range(start, stop))
            # we will assume for the memory sake that the experiment is continious
            start = experiment[0][0]
            stop = experiment[0][1]

            if clf_base == "lr":
                clf = LogisticRegression()
            elif clf_base == "sdg":
                clf = sklearn.linear_model.SGDClassifier(loss='log', penalty="l2",alpha=0.005, n_iter=5, shuffle=True)
            else:
                clf = SVC(kernel='linear', C=1)

            if action == "classify":

                if test_filename is not None:
                    scores = run_train_test_classifier(w2v_data, y_data, train_data_end, start, stop, clf=clf)

                    #scores = run_train_test_classifier(w2v_data[0:train_data_end, start:stop], y_data[0:train_data_end],
                    #                                   w2v_data[train_data_end:, start:stop], y_data[train_data_end:], clf=clf)
                else:
                    scores = run_cv_classifier(w2v_data[:, start:stop], y_data, clf=clf, n_trials=10, n_cv=3)
                print name, np.mean(scores, axis=0), scores.shape

                for i, score in enumerate(scores):
                    f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %f, %f, %f, %f, %i, %s \n" %
                           (n_trial, i, name, size, p, thresh, w2v_data.shape[0], w2v_data.shape[0]*(p+thresh-p+thresh),
                            score[0], score[1], score[2], score[3], score[4], n_components, clf_base))
                f.flush()

            elif action == "explore":

                print np.bincount(y_data)
                explore_classifier(w2v_data[:, start:stop], y_data, clf=clf, n_trials=1, orig_data=zip(x_data, ids))

            elif action == "save":

                ioutils.save_liblinear_format_data (dataname + name+"_libl.txt", w2v_data[:, start:stop], y_data)



def w2v_cluster_tweet_vocab(filename, window=0, size=0, dataname="", n_components=0, min_count=1,
                            rebuild=False):

    print "Clustering"
    x_data, y_data, stoplist, _ = make_x_y(filename, ["text"])
    w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in x_data])

    #w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
    #                            rebuild=rebuild, explore=False)

    dpgmm = transformers.DPGMMClusterModel(w2v_model=None, n_components=n_components, dataname=dataname,
                                           stoplist=stoplist, recluster_thresh=0, no_above=0.9, no_below=5,
                                           alpha=5)
    dpgmm.fit(w2v_corpus)

    #print dpgmm.dpgmm.precs_.shape


def check_w2v_model(filename="", w2v_model=None, window=0, size=0, min_count=1, dataname="", rebuild=True):

    print "Checking model for consistency"

    if w2v_model is None:
        x_data, y_data, stoplist = make_x_y(filename, ["text"])

        w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in x_data])

        logging.info("Pre-processing 2 done")
        logging.info("First line: %s" % w2v_corpus[0])
        logging.info("Last line: %s" % w2v_corpus[-1])

        w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
                                    rebuild=rebuild, explore=False)

    test_words = open("/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tests.txt", 'r').readlines()
    for word_list in test_words:
        pos_words = word_list.split(':')[0].split()
        neg_words = word_list.split(':')[1].split()
        list_similar = w2v_models.test_word2vec(w2v_model, word_list=pos_words, neg_list=neg_words)
        print "%s - %s" % (pos_words, neg_words)
        for word, similarity in list_similar:
            print similarity, repr(word)


def print_tweets(filename):

    ioutils.clean_save_tweet_text(filename, ["id_str"])

    #data = ioutils.KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    #for row in data:
    #    print row[data.text_pos]


def plot_scores(dataname):
    #if dataname == "sentiment_cv":
#    plotutils.plot_diff1_dep(dataname, withold=False)
    #    plotutils.plot_tweet_sentiment(dataname)
    #else:
    plotutils.plot_tweet_sentiment(dataname)


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--test', action='store', dest='test_filename', default="", help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')
    parser.add_argument('--min', action='store', dest='min', default='1', help='Number of LDA topics')
    parser.add_argument('--nclusters', action='store', dest='nclusters', default='30', help='Number of LDA topics')
    parser.add_argument('--clusthresh', action='store', dest='clusthresh', default='0', help='Threshold for reclustering')
    parser.add_argument('--p', action='store', dest='p', default='1', help='Fraction of labeled data')
    parser.add_argument('--thresh', action='store', dest='thresh', default='0', help='Fraction of unlabelled data')
    parser.add_argument('--ntrial', action='store', dest='ntrial', default='0', help='Number of the trial')
    parser.add_argument('--clfbase', action='store', dest='clfbase', default='lr', help='Number of the trial')
    parser.add_argument('--clfname', action='store', dest='clfname', default='w2v', help='Number of the trial')
    parser.add_argument('--action', action='store', dest='action', default='plot', help='Number of the trial')
    parser.add_argument('--rebuild', action='store_true', dest='rebuild', help='Number of the trial')
    parser.add_argument('--exp_num', action='store', dest='exp_nums', nargs='+', help='Experiments to save')
    parser.add_argument('--diff1_max', action='store', dest='diff1_max', default='5', help='Diff 1 max')
    parser.add_argument('--diff0_max', action='store', dest='diff0_max', default='1', help='Diff 0 max')


    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=arguments.dataname+"_log.txt")

    # parameters for w2v model
    min_count = int(arguments.min)
    n_components = int(arguments.nclusters)
    size=int(arguments.size)
    window=int(arguments.window)
    recluster_thresh=int(arguments.clusthresh)

    # parameters for large datasets
    ntrial = int(arguments.ntrial)
    threshhold = float(arguments.thresh)
    percentage = float(arguments.p)

    if arguments.test_filename != "":
        test_filename = arguments.test_filename
    else:
        test_filename = None

    if arguments.exp_nums:
        exp_nums = [int(n) for n in arguments.exp_nums]
    else:
        exp_nums = None


    # runs a classification experiement a given file
    if arguments.action == "classify" or arguments.action == "explore" or arguments.action == "save":
        if arguments.clfname == "w2v":

            if len(arguments.filename) > 1:
                tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                                 p=percentage, thresh=threshhold, n_trial=ntrial, min_count=min_count,
                                 clf_name=arguments.clfname, unlabeled_filenames=arguments.filename[1:],
                                 clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                                 rebuild=arguments.rebuild, action=arguments.action,
                                 experiment_nums=exp_nums,
                                 diff1_max=int(arguments.diff1_max), diff0_max=int(arguments.diff0_max))
            else:
                tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                                 p=percentage, thresh=threshhold, n_trial=ntrial, min_count=min_count,
                                 clf_name=arguments.clfname, unlabeled_filenames=None,
                                 clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                                 rebuild=arguments.rebuild, action=arguments.action,
                                 experiment_nums=exp_nums, test_filename=test_filename,
                                 diff1_max=int(arguments.diff1_max), diff0_max=int(arguments.diff0_max))
        elif arguments.clfname == 'bow':
                tweet_bow_classification(arguments.filename[0], dataname=arguments.dataname,
                                 p=percentage, thresh=threshhold, n_trial=ntrial,
                                 clf_name=arguments.clfname,
                                 clf_base=arguments.clfbase,
                                 action=arguments.action)

    # clusters the vocabulary of a given file accoding to the w2v model constructed on the same file
    elif arguments.action == "cluster":
        w2v_cluster_tweet_vocab(arguments.filename[0],
                                size=size,
                                window=window,
                                dataname=arguments.dataname,
                                n_components=n_components,
                                rebuild=arguments.rebuild,
                                min_count=min_count)

    # given a testfile of words, print most similar word from the model constructed on the file
    elif arguments.action == "check":

        check_w2v_model(filename=arguments.filename[0],
                        size=size,
                        window=window,
                        dataname=arguments.dataname,
                        min_count=min_count,
                        rebuild=arguments.rebuild)

    # plot results of a classification experiment for a certain dataname
    elif arguments.action == "plot":
        print "plot"
        plot_scores(arguments.dataname)

    # merge a unlabeled dataset, with positive labels to produce a positively labeled dataset
    elif arguments.action == "make_labels":
        ioutils.make_positive_labeled_kenyan_data(arguments.dataname)

if __name__ == "__main__":
    __main__()