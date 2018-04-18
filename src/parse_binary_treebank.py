import numpy as np
import gensim
from gensim.corpora.csvcorpus import CsvCorpus
from pandas import read_csv
import gzip
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces
import re
import read_text as rt

#In the binary case, we use the given splits of 6920 training, 872 development and 1821 test sentences.
# Likewise, in the fine-grained case, we use the standard 8544/1101/2210 splits.
# [SKIPPED] Labelled phrases that occur as subparts of the training sentences are treated as independent training instances.
# The size of the vocabulary is 15448


#SST-1: Stanford Sentiment Treebank - an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive, positive, neutral, negative, very negative), re-labeled by Socher et al. (2013). Link

# Note that data is actually provided at the phrase-level and hence we train the model on both phrases and sentences but only score on sentences at test time, as in Socher et al. (2013), Kalchbrenner et al. (2014), and Le and Mikolov (2014). Thus the training set is an order of magnitude larger than listed in the above table.

import string
from gensim.models.keyedvectors import KeyedVectors

word_vectors = {}
vocab = {}
reversed_vocab = {}


origin = "../data/sst/raw/"




def addToVocab(word, wv, id):
    if word in vocab:
        return id
    else:
        vocab[word] = id
        reversed_vocab[id] = word
        word_vectors[id] = wv
        id += 1
        return id

# Use pre-defined phrases if they match word2vec corpus, otherwise replace w/single words
# Convert to lowercase
# Retain apostrophes as they may be informative/word2vec vectors suit them
# Split into word-level arrays
def preprocessRawSentences(corpus, id):

    def addToken(sentence, word, ind):  # Add token after index
        new_s = np.empty(len(sentence) + 1, dtype=np.object)  # create new sentence with 1 extra value
        for gf in range(len(new_s)):
            if gf == ind + 1:
                new_s[gf] = word
            elif gf > ind + 1:
                new_s[gf] = sentence[gf - 1]
            else:
                new_s[gf] = sentence[gf]
        return new_s

    # batch process the sentences into tokens
    for s in range(len(corpus)):
        # In the case of words with commas or fullstops that weren't seperated, separate them
        corpus[s] = re.sub("\/", " ", corpus[s])
        corpus[s] = re.sub("\\\\", " ", corpus[s])
        corpus[s] = re.sub("\,", " ", corpus[s])
        corpus[s] = re.sub("\.", " ", corpus[s])

        corpus[s] = corpus[s].split()
        ind_to_remove = []
        w = -1
        while w+1 < len(corpus[s]):
            w += 1
            try: # if the word is already in the vocab, no need to check
                vocab[corpus[s][w]]
                continue
            except KeyError:
                nothing = 0

            punct = string.punctuation
            if corpus[s][w] == "a" or corpus[s][w] == "to" or corpus[s][w] == "and" or corpus[s][w] == "of":
                ind_to_remove.append(w)
                continue

            all_pattern = r"[{}]".format(punct)
            punct_removed = re.sub(all_pattern, "", corpus[s][w]) # remove all punct to check if its not a word
            if len(punct_removed) == 0:
                ind_to_remove.append(w)
                continue

            punct_nh = punct.replace("-", "–") # we want hyphens for phrases, but this kind are used between words not phrases
            punct_nh_na = punct_nh.replace("'", "")
            pattern = r"[{}]".format(punct_nh_na)
            corpus[s][w] = re.sub(pattern, "", corpus[s][w]) # remove all non-essential punct, e.g. commas etc
            try:
                if "'" in corpus[s][w+1] and len(corpus[s][w+1]) <= 3: # if next word contains an apostrophe and isnt its own word
                    combined_word = corpus[s][w] + corpus[s][w+1]
                    wv = checkAgainstWordVector(combined_word)
                    if len(wv) > 0: # If the combined word with the apostrophe is a real one
                        corpus[s][w] = combined_word
                        corpus[s][w+1] = ""
                        id = addToVocab(corpus[s][w], wv, id)
                        continue
            except IndexError: # if there is no next word, move onto other checks
                nothing = 0
            if "'" in corpus[s][w]:
                wv = checkAgainstWordVector(corpus[s][w])
                if len(wv) > 0:
                    id = addToVocab(corpus[s][w], wv, id)
                    continue
                # If it contains an apostrophe, and it doesn't work try using the part w/o an apostrophe
                corpus[s][w] = corpus[s][w].split("'")[0]
            if "-" in corpus[s][w]:
                wv = checkAgainstWordVector(corpus[s][w])
                if len(wv) > 0:
                    id = addToVocab(corpus[s][w], wv, id)
                    continue
                else: # split apart hyphenated phrases and add the individual words. have to do it this way,
                    #  not np.insert, because of string cutting (e.g. provoking = provoki.. don't know why it cuts...)
                    words = corpus[s][w].split("-")
                    corpus[s][w] = words[0]
                    del words[0]
                    for iw in range(0, len(words)): # add new words
                        corpus[s] = addToken(corpus[s], words[iw], w+iw)
            corpus[s][w] = re.sub(all_pattern, "", corpus[s][w])
            wv = checkAgainstWordVector(corpus[s][w])
            if len(wv) > 0:
                id = addToVocab(corpus[s][w], wv, id)
                continue
            corpus[s][w] = corpus[s][w].lower()
            wv = checkAgainstWordVector(corpus[s][w])
            if len(wv) > 0:
                id = addToVocab(corpus[s][w], wv, id)
                continue
            if len(str(re.sub('[\W_]+', "", corpus[s][w]))) == 0: # If it's a piece of punctuation not detected, or random character, remove it
                ind_to_remove.append(w)
                print("failed", corpus[s][w])
                continue
            else: # Otherwise, add it as an empty word vector
                if corpus[s][w] == "a" or corpus[s][w] == "to" or corpus[s][w] == "and" or corpus[s][w] == "of": # check if we've missed these
                    ind_to_remove.append(w)
                    continue
                test_wv = checkAgainstWordVector("movie")  # get a word vector for comparison
                wv = np.zeros(len(test_wv))
                id = addToVocab(corpus[s][w], wv, id)
                print("empty", corpus[s][w], s, w)
                continue

        corpus[s] = np.delete(corpus[s], ind_to_remove)
        #print("corpus", corpus[s] )
        #print("orig", orig_corpus[s])

    print("OK", "vocab len", len(vocab.keys()))

    numeric_corpus = []

    for s in range(len(corpus)):
        new_sentence = []
        for w in range(len(corpus[s])):
            try:
                new_sentence.append(vocab[corpus[s][w]])
            except KeyError:
                print("borked")
        numeric_corpus.append(new_sentence)

    return corpus, numeric_corpus, id

def loadSplits(folder_name):
    train = read_csv(folder_name + "train_binary_sent.csv").get_values()
    test = read_csv(folder_name + "test_binary_sent.csv").get_values()
    dev = read_csv(folder_name + "dev_binary_sent.csv").get_values()
    x_train = np.empty(len(train), dtype=np.object)
    x_test = np.empty(len(test), dtype=np.object)
    x_dev = np.empty(len(dev), dtype=np.object)
    y_train = np.empty(len(train), dtype=np.object)
    y_test = np.empty(len(test), dtype=np.object)
    y_dev = np.empty(len(dev), dtype=np.object)

    for i in range(len(train)):
        y_train[i] = train[i][0]
        x_train[i] = train[i][1]

    for i in range(len(test)):
        y_test[i] = test[i][0]
        x_test[i] = test[i][1]

    for i in range(len(dev)):
        y_dev[i] = dev[i][0]
        x_dev[i] = dev[i][1]

    return x_train, y_train, x_test, y_test, x_dev, y_dev

def loadProcessedSplits():
    x_train = np.load(origin + "x_train_n.npy")
    x_test = np.load(origin + "x_test_n.npy")
    x_dev = np.load(origin + "x_dev_n.npy")

    y_train = np.load(origin+"y_train.npy")
    y_test = np.load(origin+"y_test.npy")
    y_dev = np.load(origin+"y_dev.npy")

    return x_train, x_test, x_dev, y_train, y_test, y_dev

def loadVocabs():
    reversed_vocab = rt.load_dict(origin + "reversed_vocab.dict")
    vocab = rt.save_dict(origin + "vocab.dict")
    word_vectors = rt.save_dict(origin + "wv_vocab.dict")
    return vocab, reversed_vocab, word_vectors

if __name__ == '__main__':
    all_vectors = KeyedVectors.load_word2vec_format(
        "../data/sentiment/wordvectors/data/google/GoogleNews-vectors-negative300.bin", binary=True)

    def checkAgainstWordVector(word):
        try:
            return all_vectors.get_vector(word)
        except KeyError:
            return []


    x_train, y_train, x_test, y_test, x_dev, y_dev = loadSplits("../data/sentiment/sst_binary/")
    id = 0
    x_train_p, x_train_n, id = preprocessRawSentences(x_train, id) # Share ID across each split
    x_test_p, x_test_n, id = preprocessRawSentences(x_test, id)
    x_dev_p, x_dev_n, id = preprocessRawSentences(x_dev, id)

    if len(np.unique(list(vocab.keys()))) != len(np.unique(list(vocab.values()))):
        print("Vocab failed", len(np.unique(list(vocab.keys()))), len(np.unique(list(vocab.values()))))
        exit()
    else:
        print("Vocab succeeded", len(np.unique(list(vocab.keys()))), len(np.unique(list(vocab.values()))))

    rt.save_dict(reversed_vocab, origin+"reversed_vocab.dict")
    rt.save_dict(vocab, origin+"vocab.dict")
    rt.save_dict(word_vectors, origin+"wv_vocab.dict")

    np.save(origin+"x_train_w.npy", x_train_p)
    np.save(origin+"x_test_w.npy", x_test_p)
    np.save(origin+"x_dev_w.npy", x_dev_p)

    y_train_a = []
    y_test_a = []
    y_dev_a = []

    for y in y_train:
        y_train_a.append(int(y))
    for y in y_test:
        y_test_a.append(int(y))
    for y in y_dev:
        y_dev_a.append(int(y))

    np.save(origin+"y_train.npy", y_train_a)
    np.save(origin+"y_test.npy", y_test_a)
    np.save(origin+"y_dev.npy", y_dev_a)

    np.save(origin+"x_train_n.npy", x_train_n)
    np.save(origin+"x_test_n.npy", x_test_n)
    np.save(origin+"x_dev_n.npy", x_dev_n)

    print(x_train_n[0])
    print(x_test_n[0])
    print(x_dev_n[0])
    print(x_train_p[0])
    print(x_test_p[0])
    print(x_dev_p[0])