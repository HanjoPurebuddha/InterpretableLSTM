from sklearn.datasets import fetch_20newsgroups

import numpy as np
import string
import re
from gensim.models.keyedvectors import KeyedVectors
import read_text as rt
from keras.utils import to_categorical



word_vectors = {}
vocab = {}
reversed_vocab = {}

origin = "../data/newsgroups/raw/"

def addToVocab(word, wv, id):
    if word in vocab:
        return id
    else:
        vocab[word] = id
        reversed_vocab[id] = word
        word_vectors[id] = wv
        id += 1
        return id


def checkAgainstWordVector(word):
    try:
        return all_vectors.get_vector(word)
    except KeyError:
        return []

def loadProcessedSplits():
    x_train = np.load(origin + "x_train_n.npy")
    x_test = np.load(origin + "x_test_n.npy")

    y_train = np.load(origin+"y_train.npy")
    y_test = np.load(origin+"y_test.npy")

    return x_train, x_test, y_train, y_test

def loadVocabs():
    reversed_vocab = rt.load_dict(origin + "reversed_vocab.dict")
    vocab = rt.save_dict(origin + "vocab.dict")
    word_vectors = rt.save_dict(origin + "wv_vocab.dict")
    return vocab, reversed_vocab, word_vectors

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
        print("corpus", corpus[s] )
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





if __name__ == '__main__':

    y_train = np.load(origin+"y_train.npy")
    y_test = np.load(origin+"y_test.npy")

    train = fetch_20newsgroups(subset='train', shuffle=False, remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset='test', shuffle=False, remove=("headers", "footers", "quotes"))

    y_train = train.target
    y_test = test.target
    id = 0

    y_train_a = []
    y_test_a = []

    for y in y_train:
        y_train_a.append(int(y))
    for y in y_test:
        y_test_a.append(int(y))

    y_train_a = to_categorical(np.asarray(y_train_a))
    y_test_a = to_categorical(np.asarray(y_test_a))

    np.save(origin+".npy", y_test_a)
