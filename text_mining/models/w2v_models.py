__author__ = 'verasazonova'

import numpy as np

from text_mining.utils import textutils as tu
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec, Doc2Vec
import os.path
from random import shuffle

import logging


# **************** W2V relating functions ******************************

def make_w2v_model_name(dataname, size, window, min_count):
    return "w2v_model_%s_%i_%i_%i" % (dataname, size, window, min_count)

def make_dpgmm_model_name(dataname, n_components, n_above=0, n_below=0, alpha=5):
    return "dpgmm_model_%s_%i_%i_%.1f_%.0f" % (dataname, n_components, alpha, n_above, n_below)

def load_w2v(w2v_model_name):
    if os.path.isfile(w2v_model_name):
        w2v_model = Word2Vec.load(w2v_model_name)
        logging.info("Model %s loaded" % w2v_model_name)
        return w2v_model
    return None


def build_word2vec(text_corpus, size=100, window=10, min_count=2, dataname="none", shuffle=False,
                   alpha=0.05, sg=1, sample=1e-3, iter=15):
    """
    Given a text corpus build a word2vec model
    :param size:
    :param window:
    :param dataname:
    :return:
    """

    #w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=0.025, window=window, min_count=min_count, iter=20,
    #                     sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, negative=1e-4, cbow_mean=0)

    if shuffle:
        w2v_model = Word2Vec(size=size, alpha=alpha, window=window, min_count=min_count, iter=1,
                             sample=sample, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=sg, cbow_mean=0)
        w2v_model.build_vocab(text_corpus)
        w2v_model.iter = 1
        for epoch in range(iter):
            perm = np.random.permutation(text_corpus.shape[0])
            w2v_model.train(text_corpus[perm])
    else:
        w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=alpha, window=window, min_count=min_count, iter=iter,
                             sample=sample, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=sg, cbow_mean=0)

    logging.info("%s" % w2v_model)

    return w2v_model


# test the quality of the w2v model by extracting mist similar words to ensemble of words
def test_word2vec(w2v_model, word_list=None, neg_list=None):
    if word_list is None or not word_list:
        return []
    else:
        pos_list_checked = [word for word in word_list if word in w2v_model]
        neg_list_checked = [word for word in neg_list if word in w2v_model]
        if pos_list_checked and neg_list_checked:
            list_similar = w2v_model.most_similar_cosmul(positive=pos_list_checked, negative=neg_list_checked, topn=10)
        elif pos_list_checked:
                list_similar = w2v_model.most_similar_cosmul(positive=pos_list_checked)
        else:
            list_similar = []
        return list_similar
#------------------------------


def make_d2v_model_name(dataname, size, window, type_str):
    return "d2v_model_%s_%s_%i_%i" % (dataname, type_str, size, window)


def build_doc2vec(text_corpus, size=100, window=10, n_iter=10, alpha=0.025, min_alpha=0.001, sample=1e-3, min_count=1):
    """
    Given a text corpus build a word2vec model
    :param size:
    :param window:
    :param dataname:
    :return:
    """

    labeled_text_data = [TaggedDocument(text, ["id_"+str(line_no)]) for line_no, text in enumerate(text_corpus)]

    logging.info("Text processed")
    logging.info("Building d2v ")

    d2v_model_dm = Doc2Vec(min_count=1, window=window, size=size, sample=1e-3, negative=5, workers=4)

    #build vocab over all reviews
    d2v_model_dm.build_vocab(labeled_text_data)

    cur_alpha = alpha
    #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    for epoch in range(n_iter):

        shuffle(labeled_text_data)
        d2v_model_dm.alpha = cur_alpha
        d2v_model_dm.min_alpha = cur_alpha
        d2v_model_dm.train(labeled_text_data)
        cur_alpha -= (alpha - min_alpha) / n_iter
    return d2v_model_dm

#------------------------------