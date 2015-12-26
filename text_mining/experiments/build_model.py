import logging
import os
import pickle
import numpy as np
from text_mining.models import w2v_models, transformers
from text_mining.utils import textutils as tu

__author__ = 'verasazonova'


def build_w2v_model(w2v_corpus_list, dataname="", window=0, size=0, min_count=0, rebuild=False, explore=False):
    w2v_model_name = w2v_models.make_w2v_model_name(dataname=dataname, size=size, window=window,
                                                    min_count=min_count)
    logging.info("Looking for model %s" % w2v_model_name)
    if (not rebuild or explore) and os.path.isfile(w2v_model_name):
        w2v_model = w2v_models.load_w2v(w2v_model_name)
        logging.info("Model Loaded")
    else:
        w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in np.concatenate(w2v_corpus_list)])
        w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=min_count, dataname=dataname)
        logging.info("Model created")
    w2v_model.init_sims(replace=True)

    #check_w2v_model(w2v_model=w2v_model)
    return w2v_model


def build_dpgmm_model(w2v_corpus, w2v_model=None, n_components=0, dataname="", stoplist=None, recluster_thresh=0,
                      rebuild=False, alpha=5, no_below=6, no_above=0.9):

    model_name = w2v_models.make_dpgmm_model_name(dataname=dataname,n_components=n_components, n_below=no_below,
                                                  n_above=no_above, alpha=alpha)
    logging.info("Looking for model %s" % model_name)
    if not rebuild and os.path.isfile(model_name):
        dpgmm = pickle.load(open(model_name, 'rb'))
    else:
        dpgmm = transformers.DPGMMClusterModel(w2v_model=w2v_model, n_components=n_components, dataname=dataname,
                                               stoplist=stoplist, recluster_thresh=recluster_thresh, alpha=alpha,
                                               no_below=no_below, no_above=no_above)
        dpgmm.fit(w2v_corpus)
        pickle.dump(dpgmm, open(model_name, 'wb'))
    return dpgmm