__author__ = 'verasazonova'

import logging
import csv
import io
import numpy as np
import os.path
import text_mining.utils.textutils as tu
import sys
from operator import itemgetter
import dateutil.parser

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', errors='replace')


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


def read_tweets(filename, fields):
    """
    Read the raw csv file returns a list of tweets with certain fields

    :param filename: csv filename with a header
    :param fields: name of fields to retain
    :return: a double list of tweet data (unicode)
    """

    def fix_corrupted_csv_line(line):
        reader_str = unicode_csv_reader([line], dialect=csv.excel)
        row_str = reader_str.next()
        return row_str

    with io.open(filename, 'r', encoding='utf-8', errors='replace') as f:
        # forcing the reading line by line to avoid malformed csv entries
        reader = unicode_csv_reader(f, dialect=csv.excel)
        header = reader.next()
        # a list of indexes of fields
        field_positions = [header.index(unicode(field)) for field in fields]

        logging.info("Saving the following fields: %s " % zip(fields, field_positions))
        data = []
        try:
            for row in reader:
                if len(row) == len(header):
                    data.append([row[pos] for pos in field_positions])
                else:
                    row_str = ", ".join(row).split('\r\n')
                    data.append([fix_corrupted_csv_line(row_str[0])[pos] for pos in field_positions])
                    if len(row_str) > 1:
                        data.append([fix_corrupted_csv_line(row_str[1])[pos] for pos in field_positions])

            #data = [[row[pos] for pos in field_positions] for row in reader]
        except csv.Error as e:
            sys.exit('file %s: %s' % (filename, e))

    logging.info("Data read: %s" % len(data))
    logging.info("First line: %s" % data[0])
    logging.info("Last line: %s" % data[-1])
    return data


def clean_tweets(data, fields):
    """
    Normalzes the text of the tweets to remove newlines.
    Sorts the tweets by the date

    :param data: a double list of tweets
    :param fields: a list of field names corresponding to the data
    :return: normalized and sorted tweet list
    """

    # remove newline from text
    # make date a date
    text_str = "text"
    text_pos = None
    if text_str in fields:
        text_pos = fields.index(text_str)
    date_str = "created_at"
    date_pos = None
    if date_str in fields:
        date_pos = fields.index(date_str)
    id_str = "id_str"
    id_pos = None
    if id_str in fields:
        id_pos = fields.index(id_str)
    label_str = "label"
    label_pos = None
    if label_str in fields:
        label_pos = fields.index(label_str)

    for cnt in range(len(data)):
        if id_pos is not None:
            data[cnt][id_pos] = data[cnt][id_pos].strip()
        if text_pos is not None:
            data[cnt][text_pos] = tu.normalize_format(data[cnt][text_pos])
        if date_pos is not None:
            data[cnt][date_pos] = dateutil.parser.parse(data[cnt][date_pos])

    #sort tweets by date
    if date_pos is not None:
        data_sorted = sorted(data, key=itemgetter(date_pos))
        logging.info("Data sorted by date.  Span: %s - %s" % (data_sorted[0][date_pos], data_sorted[-1][date_pos]))
    else:
        data_sorted = data

    logging.info("Data pre-processed")
    logging.info("First line: %s" % data_sorted[0])
    logging.info("Last line: %s" % data_sorted[-1])

    return np.array(data_sorted), date_pos, text_pos, id_pos, label_pos



class KenyanCSVMessage():

    def __init__(self, filename, fields=None, stop_path="", start_date=None, end_date=None):
        """
        A class that reads and encapsulates csv twitter (or facebook) data.

        :param filename: a csv data filename.  A first row is a header explaining the fields
        :param fields: a list of field_names to include in the data
        :param stop_path: a path to a file containing the stopword list
        :return: a list of tweets sorted by date if such field exists.  newlines removed from text field (if exists)
        """
        self.filename = filename
        self.fields = fields

        if os.path.isfile(stop_path):
            logging.info("Using %s as stopword list" % stop_path)
            self.stoplist = [unicode(word.strip()) for word in
                             io.open(stop_path, 'r', encoding='utf-8').readlines()]
        else:
            self.stoplist = []

        self.data, self.date_pos, self.text_pos, self.id_pos, self.label_pos = \
            clean_tweets(read_tweets(self.filename, self.fields), self.fields)
        self.data = self.data
        if start_date is None:
            self.start_date = self.data[0][self.date_pos]
        else:
            self.start_date = start_date
        if end_date is None:
            self.end_date = self.data[-1][self.date_pos]
        else:
            self.end_date = end_date

    def __iter__(self):

        for row in self.data:
            if self.date_pos is None:
                yield row
            elif row[self.date_pos] >= self.start_date and row[self.date_pos] <= self.end_date:
                yield row


class IMDB():

    def __init__(self, dirname):
        self.name_pos = dirname + "-pos.txt"
        self.name_neg = dirname + "-neg.txt"
#        self.unlabeled_name = os.path.join(dirname, "train-unsup.txt")

        with io.open(self.name_pos, 'r', encoding='utf-8') as f:
            x_pos = [line for line in f]

        y_pos = [1 for x in x_pos]

        print self.name_pos, len(x_pos)

        x_neg = io.open(self.name_neg, 'r', encoding='utf-8').readlines()
        y_neg = [0 for x in x_neg]

        self.x = x_pos + x_neg
        self.y = y_pos + y_neg

        print len(x_neg)

    def __iter__(self):
        for x in self.x:
            yield x
