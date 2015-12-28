import logging
import os
import argparse
import pickle
from text_mining.models import w2v_models, transformers
from text_mining.utils import ioutils as io

__author__ = 'verasazonova'

# build a w2v model given a w2v corpus
def build_w2v_model(w2v_corpus, w2v_model_name="", window=0, size=0, min_count=0, rebuild=False, explore=False,
                    alpha=0.05, sg=1, iter=15, sample=1e-3):

    w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=min_count,
                                          alpha=alpha, sg=sg, sample=sample, iter=iter)
    logging.info("Model created")
    w2v_model.init_sims(replace=True)
    w2v_model.save_word2vec_format(w2v_model_name, binary=True)  #.save(w2v_model_name)
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


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')
    parser.add_argument('--min', action='store', dest='min', default='1', help='Min word occurences')
    parser.add_argument('--sample', action='store', dest='sample', default='1e-3', help='Sample frequency')
    parser.add_argument('--cbow', action='store', dest='cbow', default='0', help='cbow vs sg')
    parser.add_argument('--alpha', action='store', dest='alpha', default='0.05', help='Alpha')
    parser.add_argument('--iter', action='store', dest='iter', default='15', help='Number of iteragtions')
    parser.add_argument('--negative', action='store', dest='negative', default='0', help='negative sampling')
    parser.add_argument('--nclusters', action='store', dest='nclusters', default='30', help='Number of LDA topics')
    parser.add_argument('--clusthresh', action='store', dest='clusthresh', default='0', help='Threshold for reclustering')

    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename="log.txt")

    # parameters for w2v model
    min_count = int(arguments.min)
    size=int(arguments.size)
    window=int(arguments.window)
    alpha=float(arguments.alpha)
    sg=abs(int(arguments.cbow)-1)
    sample=int(arguments.sample)
    iter=int(arguments.iter)

    n_components = int(arguments.nclusters)
    recluster_thresh=int(arguments.clusthresh)


    naming_dict = io.get_w2v_naming()

    # build models
    w2v_corpus = io.TextFile(naming_dict["w2v_corpus"])
    w2v_model = build_w2v_model(w2v_corpus, w2v_model_name=naming_dict["w2v_model_name"],
                                window=window, size=size, min_count=min_count, sample=sample, sg=sg, alpha=alpha,
                                iter=iter)
    logging.info("Model built %s" % w2v_model)


if __name__ == "__main__":
    __main__()

