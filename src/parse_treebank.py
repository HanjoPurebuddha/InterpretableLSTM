import logging
import unittest
import os
import tempfile
import itertools
import bz2
from six import iteritems, iterkeys
from six.moves import xrange, zip as izip
from collections import namedtuple

import numpy as np

from gensim import utils, matutils
from gensim.models import doc2vec

SentimentDocument = namedtuple('SentimentDocument','words tags split sentiment')

def read_su_sentiment_rotten_tomatoes(dirname, lowercase=True):
    """
    Read and return documents from the Stanford Sentiment Treebank
    corpus (Rotten Tomatoes reviews), from http://nlp.Stanford.edu/sentiment/
    Initialize the corpus from a given directory, where
    http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    has been expanded. It's not too big, so compose entirely into memory.
    """
    logging.info("loading corpus from %s" % dirname)

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                         'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
    sentence_fixups = [(char.encode('utf-8').decode('latin1'), char) for char in chars_sst_mangled]
    # more junk, and the replace necessary for sentence-phrase consistency
    sentence_fixups.extend([
        ('Â', ''),
        ('\xa0', ' '),
        ('-LRB-', '('),
        ('-RRB-', ')'),
    ])
    # only this junk in phrases (dictionary.txt)
    phrase_fixups = [('\xa0', ' ')]

    # sentence_id and split are only positive for the full sentences

    # read sentences to temp {sentence -> (id,split) dict, to correlate with dictionary.txt
    info_by_sentence = {}
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
        with open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)  # legend
            next(splits)     # legend
            for sentence_line, split_line in izip(sentences, splits):
                (id, text) = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                (id2, split_i) = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:    # discard duplicates
                    info_by_sentence[text] = (id, int(split_i))

    # read all phrase text
    phrases = [None] * 239232  # known size of phrases
    with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
        for line in phrase_lines:
            (text, id) = line.split('|')
            for junk, fix in phrase_fixups:
                text = text.replace(junk, fix)
            phrases[int(id)] = text.rstrip()  # for 1st pass just string

    SentimentPhrase = namedtuple('SentimentPhrase', SentimentDocument._fields + ('sentence_id',))
    # add sentiment labels, correlate with sentences
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)  # legend
        for line in sentiments:
            (id, sentiment) = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            words = text.split()
            if lowercase:
                words = [word.lower() for word in words]
            (sentence_id, split_i) = info_by_sentence.get(text, (None, 0))
            split = [None,'train','test','dev'][split_i]
            phrases[id] = SentimentPhrase(words, [id], split, sentiment, sentence_id)
    """
    assert len([phrase for phrase in phrases if phrase.sentence_id is not None]) == len(info_by_sentence)  # all
    # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
    assert len([phrase for phrase in phrases if phrase.split == 'train']) == 8531  # 'train'
    assert len([phrase for phrase in phrases if phrase.split == 'test']) == 2210  # 'test'
    assert len([phrase for phrase in phrases if phrase.split == 'dev']) == 1100  # 'dev'
    """
    logging.info("loaded corpus with %i sentences and %i phrases from %s"
                % (len(info_by_sentence), len(phrases), dirname))

    return info_by_sentence
import re

data = read_su_sentiment_rotten_tomatoes("D:\datasets/treebankk\stanfordSentimentTreebank\stanfordSentimentTreebank")

sentences = np.asarray(list(data.keys()), dtype=str)



splits = np.asarray(list(data.values()), dtype=np.int8)


# set sentences to each split
for s in range(len(sentences)):
    if classes[s] == 1:

# Still need to get labels...

np.save("../data/sentiment/sst/sentences.txt", sentences)
np.save("../data/sentiment/sst/classes.txt", binary_classes)