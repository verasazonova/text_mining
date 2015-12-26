from text_mining.utils import textutils as tu

__author__ = 'verasazonova'

import logging
import os
import codecs
from os.path import basename
import dateutil.parser
import numpy as np
from text_mining.corpora.csv_tweet_reader import clean_tweets, KenyanCSVMessage, read_tweets


def read_counts_bins_labels(dataname):
    counts = np.loadtxt(dataname+"_cnts.txt")
    bin_lows = []
    with open(dataname+"_bins.txt", 'r') as f:
        for line in f:
            bin_lows.append(dateutil.parser.parse(line.strip()))
    topic_definitions = []
    with open(dataname+"_labels.txt", 'r') as f:
        for line in f:
            topic_definitions.append(line.strip().split())
    topics = []
    with codecs.open(dataname+"_labels_weights.txt", 'r', encoding='utf-8') as f:
        for line in f:
            topics.append([(tup.split(',')[1], tup.split(',')[0]) for tup in line.strip().split(' ')])

    return counts, bin_lows, topics



def clean_save_tweet_text(filename, fields):

    data, date_pos, text_pos, id_pos, label_pos = clean_tweets(read_tweets(filename, fields), fields)
    print data[0]
    print text_pos
    if "text" in fields:
        processed_data = [tu.normalize_punctuation(text[text_pos]) for text in data]

    filename_out = basename(filename).split('.')[0] + "_short_"
    if "created_at" in fields:
        filename_out += "sorted_"
    filename_out += "_".join(fields) + ".csv"

    #save sorted file with newline charactes removed from the text
    with codecs.open(filename_out, 'w', encoding='utf-8') as fout:
        # write the data
        fout.write("\",\"".join(fields)+"\n")
        for row in data:
            fout.write(" ".join(row)+"\n")


def save_liblinear_format_data(filename, x_data, y_data):

    with open(filename, 'w') as fout:
        for x, y in zip(x_data, y_data):
            fout.write("%i " % int((y-0.5)*2) )
            for i, coordinate in enumerate(x):
                fout.write("%i:%f " % (i+1, coordinate))
            fout.write("\n")

def is_politica_tweet(tweet_text, political_words):

    for word in tweet_text:
        if word in political_words:
            return True

    return False


def extract_political_tweets(filename):
    fields = ["id_str", "text", "created_at"]
    data = KenyanCSVMessage(filename, fields)
    political_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/political.txt"

    if os.path.isfile(political_path):
            logging.info("Using %s as stopword list" % political_path)
            political_list = [unicode(word.strip()) for word in
                              codecs.open(political_path, 'r', encoding='utf-8').readlines()]
    else:
        political_list = []

    with codecs.open(basename(filename).split('.')[0]+"_political.csv", 'w', encoding='utf-8') as fout:
        fout.write("\",\"".join(fields)+"\n")
        for row in data:
            if is_politica_tweet(row[data.text_pos], political_list):
                # write the data
                fout.write("\",\"".join(row)+"\n")


def make_positive_labeled_kenyan_data(dataname):
    dataset_positive = KenyanCSVMessage(dataname+"_positive.csv", fields=["id_str"])
    data_positive = [tweet[dataset_positive.text_pos] for tweet in dataset_positive]

    print len(data_positive)
    dataset = KenyanCSVMessage(dataname+".csv", fields=["text", "id_str"])

    cnt_pos = 0
    with codecs.open(dataname+"_annotated_positive.csv", 'w', encoding='utf-8') as fout:
        fout.write("id_str,text,label\n")
        for cnt, tweet in enumerate(dataset):
            if cnt % 10000 == 0:
                print cnt, cnt_pos
            if tweet[dataset.id_pos] in data_positive:
                cnt_pos += 1
                fout.write(tweet[dataset.id_pos] + ",\"" + tweet[dataset.text_pos].replace("\"", "\"\"") + "\",T\n")
            else:
                fout.write(tweet[dataset.id_pos] + ",\"" + tweet[dataset.text_pos].replace("\"", "\"\"") + "\",F\n")

    print cnt_pos


def save_positives(positives, dataname):
    with codecs.open(dataname+"_additional_positives.csv", 'w', encoding='utf-8') as fout:
        fout.write("id_str,text\n")
        for tweet, id in positives:
            fout.write(id + ",\"" + tweet.replace("\"", "\"\"") + "\"\n")

def save_words_representations(filename, word_list, vec_list):
    # saving word representations
    # word, w2v vector
    with codecs.open(filename, 'w', encoding="utf-8") as fout:
        for word, vec in zip(word_list, vec_list):
            fout.write(word + "," + ",".join(["%.8f" % x for x in vec]) + "\n")


# save cluster information: size and central words
def save_cluster_info(filename, cluster_info):
    with codecs.open(filename, 'w', encoding="utf-8") as fout:
        for cluster_dict in cluster_info:
            fout.write("%2i, %5i,   : " % (cluster_dict['cnt'], cluster_dict['size']))
            for j, word in enumerate(cluster_dict['words']):
                if j < 10:
                    fout.write("%s " % word)
            fout.write("\n")
