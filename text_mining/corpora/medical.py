#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:22:19 2014

@author: vera
"""

import argparse
import os.path
import codecs
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim import corpora
from gensim.models.doc2vec import LabeledSentence

def readarticles(filename, article_fields):
    article_list = []
    article_dict = {}
    dataset = os.path.basename(filename).split('.')[0].split('_')[1]
    with open(filename, 'r') as f:
        for line in f:
            # print line
            m = re.search('\*\*\*\*\*\* (.+)', line.strip())
            if m:
                if not (article_dict == {}):
                    article_list.append(article_dict)
                    # reinitialize the out string
                #article_dict = {'id': dataset  + "_" + m.group(1), 'A': '', 'T': '', 'M': '' }
                article_dict = {'id': '_*_'+m.group(1), 'A': '', 'T': '', 'M': '' }
            else:
                m = re.search('----([K|T|A|P|M]) (.+)', line.strip())
                if m:
                    # if one of the valid options
                    if m.group(1) == 'K':
                        # the full id of the article is the filename joined by the id of the article
                        article_dict['class'] = m.group(2)
                    if m.group(1) in article_fields:
                        # if in the middle of the article, and the fields are one of the valid ones -
                        # create the out dictionary
                        for field in article_fields:
                            if m.group(1) == field:
                                # for separate fields - copy the field
                                str_field = m.group(2)
                                article_dict[field] = str_field.replace("\"", "\\\"")

        # process the write out string for the last article
        article_list.append(article_dict)
    return article_list


def normalize(phrase):
    norm_phrase = phrase.lower().replace('<br>', ' ').replace('<br \/', ' ')
    for punctuation in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/', '\"', '*', '^', '%', '$', '&', '<', '>', '-']:
        norm_phrase = norm_phrase.replace(punctuation, ' ' + punctuation +' ')
    return norm_phrase

def writearticles(articles, filename):
    print len(articles)
    print filename
    with open(filename, 'w') as fout:
        for article in articles:
            fout.write("%s %s\n" % (article['id'], normalize(article['T'] + article['A'])))

def writearticles_posneg(articles, filename):
    print len(articles)
    print filename
    with open(filename+'-pos.txt', 'w') as fout_pos:
        with open(filename+'-neg.txt') as fout_neg:
            for article in articles:
                if article['class'] == 'I':
                    fout_pos.write("%s\n" % (article['T'] + article['A']))
                elif article['class'] == 'E':
                    fout_neg.write("%s\n" % (article['T'] + article['A']))


stop_filename = "stopwords.txt"
stop_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), stop_filename)
print stop_path
if os.path.isfile(stop_path):
    print "Using stopwords.txt as stopword list"
    stop = [word.strip() for word in open(stop_path, 'r').readlines()]
else:
    print "Using nltk stopwords"
    stop = stopwords.words('english')
#    stop = set(['a', 'the'])

def word_valid(word):
    if (word in stop) or len(word) < 2: #or re.match('\d+([,.]\d)*', word) or re.match(r".*\\.*", word) \
            #or re.match(r"\W+", word):
        return False
    return True


def word_tokenize(text):
    tokens = [unicode(word.translate(None, '!?.,;:\'\"'), errors='replace') for word in text.translate(None, '()[]').lower().split() if
              word_valid(word)]
    return tokens


#def sent_tokenize(text):
#    tokens = [sent for sent in text.split('. ')]
#    tokens = nl
#    return tokens


def mesh_tokenize(text):
    tokens = []
    for word in text.lower().split():
        m = re.match("([sm]_)+(.*)_mesh", word)
        if m:
            tokens.append(m.group(2))
    return tokens


class AugmentedCorpus:
    def __init__(self, filename):
        self.filename = filename
        self.target = []

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                self.target.append(int(line.strip().split(',')[1]))
                yield line.strip().split(',')[0].split()

    def get_target(self):
        if self.target:
            return self.target
        else:
            print "you must iterated the class first"


class PMCOpenSubset:
    def __init__(self, filename, labeled=True, stop_path=""):
        self.labeled = labeled
        if os.path.isfile(stop_path):
            self.stoplist = [unicode(word.strip()) for word in
                             codecs.open(stop_path, 'r', encoding='utf-8').readlines()]
        else:
            self.stoplist = []
        self.filename = filename

    def __iter__(self):
        with codecs.open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                yield line


class MedicalReviewAbstracts:
    def __init__(self, filenames, article_fields, labeled=True, tokenize=True, stop_path=""):
        if type(filenames) is not list:
            filenames = [filenames]
        print filenames
        self.articles = []
        for file in filenames:
            self.articles += readarticles(file, article_fields)
        #self.dataset = os.path.basename(filename)
        self.article_fields = article_fields
        self.labeled = labeled
        self.tokenize = tokenize
        if os.path.isfile(stop_path):
            self.stoplist = [unicode(word.strip()) for word in
                             codecs.open(stop_path, 'r', encoding='utf-8').readlines()]
        else:
            self.stoplist = []


    def __iter__(self):
        for article in self.articles:
            if self.tokenize:
                text_tokens = []
                mesh_tokens = []
                if ('T' in article) and ('A' in article):
                    text_tokens = word_tokenize(article['T'] + article['A'])
                elif 'A' in article:
                    text_tokens = word_tokenize(article['A'])
                elif 'T' in article:
                    text_tokens = word_tokenize(article['T'])

                if 'M' in article:
                    mesh_tokens = mesh_tokenize(article['M'])
            else:
                text_tokens = ""
                mesh_tokens = ""
                for key in self.article_fields:
                    if key in article:
                        text_tokens += article[key]

            if self.labeled:
                yield LabeledSentence( text_tokens + mesh_tokens, [article['id']] )
            else:
                yield text_tokens + mesh_tokens

    def get_target(self):
        return [1 if article['class'] == 'I' else 0 for article in self.articles]


    def print_statistics(self):

        n = len(self.articles)
        pos_ind = [i for i in range(n) if self.articles[i]['class'] == 'I' ]
        n_pos = len(pos_ind)
        #print "Dataset, Percent positives, # positives, # total: "
        return self.dataset, n_pos*100.0 / n, n_pos, n




def print_stats(mra):
    name, p_pos, n_pos, n = mra.print_statistics()
    x = [article for article in mra]
    y = mra.get_target()
    dictionary = corpora.Dictionary(x)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    n_w =  len(dictionary)
    x_pos = [article for i, article in enumerate(mra) if y[i] == 1]
    dictionary_pos = corpora.Dictionary(x_pos)
    dictionary_pos.filter_extremes(no_below=2, no_above=0.9)
    n_w_pos = len(dictionary_pos)
    print ", ".join(map(str, [name, p_pos, n_pos, n, n_w, n_w_pos]))


def get_filename(dataset):
    prefix = os.environ.get("MEDAB_DATA")
    return prefix + "/units_" + dataset + ".txt"

def prep_arguments(arguments):

    prefix = "/Users/verasazonova/no-backup/medab_data" #os.environ.get("MEDAB_DATA")
    datasets = []
    filenames = []
    if (arguments.filename is None) and (arguments.dataset is None):
        datasets = ["Estrogens"]
        filenames = [prefix + "/units_Estrogens.txt"]
    elif arguments.filename is None:
        datasets = arguments.dataset
        print datasets, prefix
        filenames =  [prefix + "/units_" + dataset + ".txt" for dataset in datasets]
    else:
        exit()

    return datasets, filenames


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', nargs='+', dest='filename', help='Data filename')
    parser.add_argument('-d', action='store', nargs="+", dest='dataset', help='Dataset name')
    arguments = parser.parse_args()

    datasets, filenames = prep_arguments(arguments)
    mra = MedicalReviewAbstracts(filenames, ['T', 'A'], labeled=False, tokenize=False)
#        print_stats(mra)
    for article in mra:
        print article

    #print arguments.filename
    #writearticles(readarticles(arguments.filename[0], ['T', 'A']), "all_abstracts_id.txt")

if __name__ == "__main__":
    __main__()