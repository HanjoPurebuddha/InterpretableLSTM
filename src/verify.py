import numpy as np
import read_text as rt
import parse_binary_treebank as pbt
import string
import re

origin = "../data/sst/raw/"

vocab = rt.load_dict(origin+"vocab.npy")
wv = rt.load_dict(origin+"wv_vocab.npy")
x_train_w = np.load(origin+"x_train_w.npy")
x_test_w = np.load(origin+"x_test_w.npy")
x_dev_w = np.load(origin+"x_dev_w.npy")


reversed_vocab = rt.load_dict(origin+"reversed_vocab.dict")

x_train, y_train, x_test, y_test, x_dev, y_dev = pbt.loadSplits("../data/sentiment/sst_binary/")

def verifyIndexed(indexed, words, r_vocab):
    print("verifying index")
    for s in range(len(indexed)):
        for w in range(len(indexed[s])):
            if r_vocab[indexed[s][w]] != words[s][w]:
                print(">>>>> Failed indexed equivalence")
                print(indexed[s])
                print(words[s])
                break

def verifySentence(words, orig_corpus):
    print("verifying sentence")
    punct = string.punctuation + "–" # add characters present in the original corpus that arent in string.punct
    pattern = r"[{}]".format(punct)
    for s in range(len(orig_corpus)):
        orig_corpus[s] = re.sub(pattern, " ", orig_corpus[s])
        orig_corpus[s] = orig_corpus[s].replace(' a ', ' ')
        orig_corpus[s] = orig_corpus[s].replace(' of ', ' ')
        orig_corpus[s] = orig_corpus[s].replace(' to ', ' ')
        orig_corpus[s] = orig_corpus[s].replace(' and ', ' ')
        orig_corpus[s] = orig_corpus[s].replace(' ', '') # remove whitespace
        orig_corpus[s] = orig_corpus[s].lower()
        combined_sent = "".join(words[s]).lower()
        combined_sent = re.sub(pattern, "", combined_sent)
        if orig_corpus[s] != combined_sent:
            print(">>>>> Failed sentence equivalence")
            print(orig_corpus[s])
            print(words[s])

#verifyIndexed(x_train_n, x_train_w, reversed_vocab)
#verifyIndexed(x_test_n, x_test_w, reversed_vocab)
#verifyIndexed(x_dev_n, x_dev_w, reversed_vocab)

verifySentence(x_train_w, x_train)
print("DONE")