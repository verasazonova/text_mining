__author__ = 'verasazonova'

import logging
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import cross_validation, grid_search
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn
import pickle
from text_mining.utils import ioutils as io


def explore_classifier(x, y, clf=None, n_trials=1, orig_data=None):
    false_positives = []
    false_negatives = []
    positives = []
    print np.bincount(y)
    for n in range(n_trials):
        skf = cross_validation.StratifiedKFold(y, n_folds=5)  # random_state=n, shuffle=True)
        predictions = cross_validation.cross_val_predict(clf, x, y=y, cv=skf, n_jobs=1, verbose=2)

        print len(predictions)
        print np.bincount(predictions)
#        x_train, x_test, y_train, y_test, _, orig_test = train_test_split(x, y, orig_data, test_size=0.2, random_state=n)
#        #print x_train[0]
#        clf.fit(x_train, y_train)
#        y_pred = clf.predict(x_test)

        print(sklearn.metrics.confusion_matrix(y, predictions))
        print(sklearn.metrics.classification_report(y, predictions))
        print(sklearn.metrics.precision_score(y, predictions))
        print
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                if y[i] == 1:
                    false_negatives.append(orig_data[i])
                else:
                    false_positives.append(orig_data[i])
            if y[i] == 1:
                positives.append(orig_data[i])

    print len(false_negatives), len(false_positives)
    false_positives = list(set(false_positives))
    false_negatives = list(set(false_negatives))
    print "False positives:", len(false_positives)
    io.save_positives(false_positives, dataname="false")
    io.save_positives(false_negatives, dataname="false_negatives")
    io.save_positives(positives, dataname="positives")
    print
    print "False negatives:", len(false_negatives)


def run_cv_classifier(x, y, clf=None, fit_parameters=None, n_trials=10, n_cv=2, direct=False):
    # all cv will be averaged out together
    scores = np.zeros((n_trials * n_cv, 5))
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))

        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        if direct:
            n_fold = 0
            for train_ind, test_ind in skf:
                clf.fit(x_shuffled[train_ind, :], y_shuffled[train_ind])
                predictions = clf.predict(x_shuffled[test_ind, :])
                for i, metr in enumerate([sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
                                            sklearn.metrics.recall_score, sklearn.metrics.f1_score,
                                            sklearn.metrics.roc_auc_score]):

                    sc = metr(y_shuffled[test_ind],predictions)
                    scores[n * n_cv + n_fold, i] = sc
                n_fold += 1
        else:
            for i, scoring_name in enumerate(['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
                sc = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring=scoring_name,
                                                                           verbose=0, n_jobs=1)
                scores[n * n_cv: (n+1) *n_cv, i] = sc
    #print scores, scores.mean(), scores.std()
    return scores


def run_train_test_classifier(x_train, y_train, x_test, y_test, start, stop, clf=None):
    #print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    scores = np.zeros((1, 5))
    MAX = 2e6

    train_end = len(x_train)
    # if we can fit the whole array in memory.


    if train_end < MAX:
        clf.fit(csr_matrix(x_train[:, start:stop]), y_train)
    # if not, go by batches.

    else:
        batch_size = 10000
        n_batches = int(train_end/batch_size)
        all_classes = np.unique(y_train)
        print "Learning by batches: %i " % n_batches
        logging.info("Learning by batches: %i " % n_batches )

        # cycle over the data 5 times, shuffling the order of training
        for r in range(5):
            inds = shuffle(range(train_end), random_state=r)
            for i in range(n_batches):
                logging.info("Run  %i %i " % (r, i))
                batch_inds = inds[i*batch_size:(i+1)*batch_size]
                clf.partial_fit(csr_matrix(x_train[batch_inds, start:stop]), y_train[batch_inds], classes=all_classes)

            # last batch
            batch_inds = inds[n_batches*batch_size:]
            if batch_inds:
                clf.partial_fit(csr_matrix(x_train[batch_inds, start:stop]), y_train[batch_inds], classes=all_classes)

    predictions = clf.predict(csr_matrix(x_test[:, start:stop]))
    for i, metr in enumerate([sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
                              sklearn.metrics.recall_score, sklearn.metrics.f1_score, sklearn.metrics.roc_auc_score]):
        sc = metr(y_test, predictions)
        scores[0, i] = sc

    return scores


def run_grid_search(x, y, clf=None, parameters=None, fit_parameters=None):
    if clf is None:
        raise Exception("No classifier passed")
    if parameters is None:
        raise Exception("No parameters passed")
    print parameters
    grid_clf = grid_search.GridSearchCV(clf, param_grid=parameters, fit_params=fit_parameters,
                                        scoring='accuracy',
                                        iid=False, cv=2, refit=True)
    grid_clf.fit(x, y)
    print grid_clf.grid_scores_
    print grid_clf.best_params_
    print grid_clf.best_score_

    return grid_clf.best_score_


def build_experiments(feature_crd, names_orig=None, experiment_nums=None, start=0):
    if names_orig is None:
        names_orig = sorted(feature_crd.keys())
    experiments = []
    print "Building experiments: ", experiment_nums
    names = []
    for name in names_orig[start:]:
        if (experiment_nums is None) or (int(name[:2]) in experiment_nums):
            names.append(name)
            experiments.append( [(feature_crd[names_orig[start]][0], feature_crd[name][1])])
    return names, experiments

def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('--clfbase', action='store', dest='clfbase', default='lr', help='Base Classifier name')
    parser.add_argument('--type', action='store', dest='type', default='classify', help='CV test or grid_search')
    parser.add_argument('--exp_num', action='store', dest='exp_nums', nargs='+', help='Experiments to save')
    parser.add_argument('--parameters', action='store', dest='params', nargs='+', help='Parameters to save')
    parser.add_argument('--txt', action='store_true', dest='txt', help='Data format: binary by default')
    parser.add_argument('--start', action='store', dest='start', default='0', help='Test with sent2vec')


    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename="log.txt")

    if arguments.exp_nums:
        exp_nums = [int(n) for n in arguments.exp_nums]
    else:
        exp_nums = None

    naming_dict = io.get_w2v_naming()
    dataname = arguments.dataname
    experiment_type = arguments.type
    clf_base = arguments.clfbase

    if arguments.txt:
        x_train_data = np.loadtxt(naming_dict["x_train_vec"]+".txt")
        if experiment_type == "test":
            x_test_data = np.loadtxt(naming_dict["x_test_vec"]+".txt")
        w2v_feature_crd = {'00_sentence': (1, x_train_data.shape[1])}
        print w2v_feature_crd

    else:
        x_train_data = np.load(naming_dict["x_train_vec"]+".npy")
        if experiment_type == "test":
            x_test_data = np.load(naming_dict["x_test_vec"]+".npy")
        w2v_feature_crd = pickle.load(open(naming_dict["w2v_features_crd_name"], 'rb'))

    y_train_data = np.loadtxt(naming_dict["y_train"])
    if experiment_type == "test":
        y_test_data = np.loadtxt(naming_dict["y_test"])


    names, experiments = build_experiments(w2v_feature_crd, experiment_nums=exp_nums, start=int(arguments.start))

    if clf_base == "lr":
        clf = LogisticRegression()
    elif clf_base == "sdg":
        clf = sklearn.linear_model.SGDClassifier(loss='log', penalty="l2",alpha=0.005, n_iter=5, shuffle=True)
    else:
        clf = SVC(kernel='linear', C=1)

    logging.info("Loaded a base classifier: %s" % clf)

    print "Built experiments: ", names

    logging.info("Built experiments: %s" % str(names))

    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
        for name, experiment in zip(names, experiments):
            logging.info("Experiment %s %s" % (name, str(experiment)))
            start = experiment[0][0]
            stop = experiment[0][1]
            print name, experiment, experiment_type, start, stop

            if experiment_type == "cv":
                scores = run_cv_classifier(x_train_data[:, start:stop], y_train_data, clf=clf, n_trials=3, n_cv=3)
            elif experiment_type == "test":
                scores = run_train_test_classifier(x_train_data, y_train_data, x_test_data, y_test_data, start, stop, clf=clf)

            print name, np.mean(scores, axis=0), scores.shape

            for i, score in enumerate(scores):
                f.write("%i,%s,%f, %f, %f, %f, %f, %s, %s \n" %
                       (i, name, score[0], score[1], score[2], score[3], score[4],
                        clf_base, ", ".join(arguments.params)))
            f.flush()


if __name__ == "__main__":
    __main__()

