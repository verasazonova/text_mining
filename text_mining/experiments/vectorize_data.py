import logging
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from text_mining.utils import ioutils as io
from text_mining.models import transformers
from text_mining.utils import textutils as tu
import argparse

__author__ = 'verasazonova'


def build_and_vectorize_w2v(x_data=None, y_data=None, dataname="",
                            w2v_model=None, diff1_max=3, diff0_max=1):

    # get features from models
    w2v = transformers.W2VTextModel(w2v_model=w2v_model, no_above=1.0, no_below=1,
                                    diffmax0=diff0_max, diffmax1=diff1_max)
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





def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--diff1_max', action='store', dest='diff1_max', default='5', help='Diff 1 max')
    parser.add_argument('--diff0_max', action='store', dest='diff0_max', default='1', help='Diff 0 max')

    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename="log.txt")

    # parameters for w2v model
    diff1_max=int(arguments.diff1_max)
    diff0_max=int(arguments.diff0_max)

    naming_dict = io.get_w2v_naming()

    # load w2v model
    w2v_model = Word2Vec.load_word2vec_format(naming_dict["w2v_model_name"], binary=False)

    # load x_data
    x_data = io.load_data(naming_dict["x_train"])

    w2v_data, w2v_feature_crd = build_and_vectorize_w2v(x_data=x_data, w2v_model=w2v_model,
                                                          diff1_max=diff1_max, diff0_max=diff0_max)

    # scale

    print "Vectorized.  Saving"
    logging.info("Vectorized. Saving")
    np.save(naming_dict["w2v_data_name"], np.ascontiguousarray(w2v_data))

    pickle.dump(w2v_feature_crd, open(naming_dict["w2v_features_crd_name"], 'wb'))

    logging.info("Scaling")
    print "Scaling"
    w2v_data = scale_features(w2v_data, w2v_feature_crd)
    print "Scaled. Saving"
    logging.info("Scaled. Saving")

    np.save(naming_dict["w2v_data_name"], np.ascontiguousarray(w2v_data))

    print "Building experiments"
    logging.info("Building experiments")



if __name__ == "__main__":
    __main__()

