__author__ = 'verasazonova'

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
from six import string_types
import logging
from sklearn.manifold import TSNE


def plot_words_distribution(word_vecs, n_topics, dataname=""):

    topic_vecs = np.zeros((n_topics, len(word_vecs[0])))
    for i in range(n_topics):
        topic_vecs[i] = np.sum(word_vecs[i*20:i*20+20])

    ts = TSNE(2)
    logging.info("Reducing with tsne")
    reduced_vecs = ts.fit_transform(topic_vecs)

    cmap = get_cmap(n_topics)

    plt.figure()

    for i in range(n_topics):
        plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=cmap(i), markersize=8, label=str(i))
    plt.legend()
    plt.savefig(dataname+"_words.pdf")


def get_cmap(n_colors):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Set1') #'Spectral')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def plot_tweets(counts, dates, labels, clusters, dataname):

    time_labels = [date.strftime("%m-%d") for date in dates]

    n_topics = counts.shape[0]
    n_bins = counts.shape[1]
    ind = np.arange(n_bins)
    cmap = get_cmap(n_topics)

    width = 0.35
    totals_by_bin = counts.sum(axis=0)+1e-10

    log_totals = np.log10(totals_by_bin)

    plt.figure()

    plt.subplot(211)
    plt.plot(ind+width/2., totals_by_bin)
    plt.xticks([])
    plt.ylabel("Total twits")
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.grid()

    plt.subplot(212)
    polys = plt.stackplot(ind, log_totals*counts/totals_by_bin, colors=[cmap(i) for i in range(n_topics)])

    legend_proxies = []
    for poly in polys:
        legend_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))

    plt.ylabel("Topics.  % of total twits")
    plt.xticks((ind+width/2.)[::4], time_labels[::4], rotation=60)
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.ylim([0, np.max(log_totals)])

    common_words = set(labels[0])
    for label in labels[1:]:
        common_words = common_words.intersection(set(label))

    logging.info("Words common to all labels: %s" % common_words)

    label_corpus = []

    clean_labels = []
    for i, label in enumerate(labels):
        legend = str(clusters[i]) + ", "
        legend_cnt = 0
        word_list = []
        for word in label:
            word_list.append(word)
            if word not in common_words:
                legend += word + " "
                legend_cnt += len(word) + 1
            if legend_cnt > 100:
                legend += '\n '
                legend_cnt = 0
        label_corpus.append(word_list)
        clean_labels.append(legend)

    logging.info("Saved in %s" % (dataname+".pdf"))
    plt.figlegend(legend_proxies, clean_labels, 'upper right', prop={'size': 6}, framealpha=0.5)
    plt.savefig(dataname+".pdf")


def extract_xy_average(data, xind, yind, cind, cval):
    data_c = data[data[:, cind] == cval]

    xvals = sorted(list(set(data_c[:, xind])))
    yvals = []
    yerr = []
    n = []
    for xval in xvals:
        i = data_c[:, xind] == xval
        yvals.append(data_c[i, yind].mean())
        yerr.append(data_c[i, yind].std())
        n.append(len(data_c[i, yind]))
    return np.array(xvals), np.array(yvals), np.array(yerr), n


def extract_data_series(data, xind, yind, cind, cval):
    data_c = data[data[:, cind] == cval]

    xvals = range(len(data_c))
    yvals = []
    yerr = []
    for xval in xvals:
        # using the row number
        i = xval
        yvals.append(data_c[i, yind].mean())
        yerr.append(data_c[i, yind].std())
    return np.array(xvals), np.array(yvals), np.array(yerr)


def extract_conditions(data, conditions=None):
    if conditions is None:
        return data
    data_c = data

    # a list of (ind, value) tuples or of (ind, [val1, val2, val3]) tuples
    for ind, val in conditions:
#        if isinstance(val, list):
#            tmp = []
#            for v in val:
#                tmp.append(data_c[data_c[:, ind] == val])
#                print tmp
 #           data_c = np.concatenate(tmp)
 #       else:

         data_c = data_c[data_c[:, ind] == val]

    return data_c


def plot_multiple_xy_averages(data_raw, xind, yind, cind, marker='o', cdict=None, conditions=None, witherror=False,
                              line='-',series=False, labels=None, ax=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))
    for cval in cvals:

        np.set_printoptions(precision=4)  #formatter={'float': '{: 0.3f}'.format})
        if series:
            xvals, yvals, yerrs = extract_data_series(data, xind, yind, cind, cval)
            if labels is not None and cval in labels:
                print "%-10s" % labels[cval],
            else:
                print "%-10s" % cval,
            print "%.4f +- %.4f" % (yvals.mean(), yvals.std())
        else:
            xvals, yvals, yerrs, n = extract_xy_average(data, xind, yind, cind, cval)
            print cval, n, xvals, yvals, yerrs
        # if no color is supplied use black
        if cdict is None:
            color = 'k'
        # if cdict is a string - assume it is a color
        elif isinstance(cdict, string_types):
            color = cdict
        # by default cdict is a dictionary of color-value pairs
        else:
            color = cdict[cval]

        if labels is not None and cval in labels:
            label = labels[cval]
        else:
            label = cval
        if ax is None:
            ax = plt.gca()
        if witherror:
            ax.errorbar(xvals, yvals, yerr=yerrs, fmt=line, marker=marker, color=color, label=label, elinewidth=0.3,
                        markersize=5)
        else:
            ax.plot(xvals, yvals, line, marker=marker, color=color, label=label)


def extract_base(data, xind, yind, cind, cval):

    ind = data[:, cind] == cval
    #xvals = [min(data[:, xind]), max(data[:, xind])]
    xvals = plt.xlim()
    yvals = [data[ind, yind].mean(), data[ind, yind].mean()]
    std = [data[ind, yind].std(), data[ind, yind].std()]
    print "bow", cval, xvals, yvals, std
    return xvals, yvals, std


def plot_multiple_bases(data_raw, xind, yind, cind, cdict=None, conditions=None, labels=None, ax=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))

    if ax is None:
        print "using gca"
        ax = plt.gca()
    for cval in cvals:
        xvals, yvals, std = extract_base(data, xind, yind, cind, cval)
        if cdict is None:
            color = 'k'
        elif isinstance(cdict, string_types):
            color = cdict
        else:
            color = cdict[cval]
        if labels is None or cval not in labels:
            label = cval
        else:
            label = labels[cval]
        ax.plot(xvals, yvals, '--', color=color, label=label)
        # ax.axhspan(yvals[0]-std[0], yvals[0]+std[0], facecolor=color, alpha=0.1, edgecolor='none')



