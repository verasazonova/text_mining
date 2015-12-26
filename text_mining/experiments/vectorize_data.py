import logging
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from text_mining.experiments.build_model import build_w2v_model
from text_mining.models import transformers
from text_mining.utils import textutils as tu

__author__ = 'verasazonova'


def build_and_vectorize_w2v(x_data=None, y_data=None, unlabeled_data=None, window=0, size=0, dataname="",
                        rebuild=False, action="classify", stoplist=None, min_count=1,
                        diff1_max=3, diff0_max=1):

    w2v_corpus = [x_data, unlabeled_data]
    if action == "explore":
        explore = True
    else:
        explore = False

    logging.info("Classifying %s, %i, %i" % (dataname, len(w2v_corpus), min_count,))
    # build models
    w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
                                rebuild=rebuild, explore=explore)

    # get features from models
    w2v = transformers.W2VTextModel(w2v_model=w2v_model, no_above=1.0, no_below=1, diffmax0=diff0_max, diffmax1=diff1_max)

    # get matrices of features from x_data
    w2v_data = w2v.fit_transform(x_data)

    print w2v_data.shape

    return w2v_data, w2v.feature_crd


def build_and_vectorize_dpgmm(x_data=None, y_data=None, unlabeled_data=None, dataname="", n_components=0,
                        rebuild=False, action="classify", stoplist=None, min_count=1, recluster_thresh=0,
                        no_above=0.9, no_below=5):

    w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in np.concatenate([x_data, unlabeled_data])])

    dpgmm = transformers.DPGMMClusterModel(w2v_model=None, n_components=n_components, dataname=dataname,
                                           stoplist=stoplist, recluster_thresh=0, no_above=no_above, no_below=no_below,
                                           alpha=5)

    pickle.dump(dpgmm, open(dataname+"_dpgmm", 'wb'))

    dpgmm.fit(w2v_corpus)
    dpgmm_data = dpgmm.transform(x_data)

    pickle.dump(dpgmm_data, open(dataname+"_dpgmm_data", 'wb'))


    return dpgmm_data, dpgmm.feature_crd


def scale_features(data, feature_crd):
    # scale features
#    for name, (start, end) in feature_crd.items():
#        data[:, start:end] = StandardScaler().fit_transform(data[:, start:end])

    data = StandardScaler(copy=False).fit_transform(data)
    print "scaled"
    return data


def build_experiments(feature_crd, names_orig=None, experiment_nums=None):
    if names_orig is None:
        names_orig = sorted(feature_crd.keys())
    experiments = []
    print "Building experiments: ", experiment_nums
    names = []
    for name in names_orig:
        if (experiment_nums is None) or (int(name[:2]) in experiment_nums):
            names.append(name)
            experiments.append( [(0, feature_crd[name][1])])
    return names, experiments