import numpy as np
import parse_binary_treebank as ptb
import read_text as rt
from keras.datasets import imdb

""" PTB
x_train, x_test, x_dev, y_train, y_test, y_dev = ptb.loadProcessedSplits()
vocab, reversed_vocab, wv = ptb.loadVocabs()
"""
index_from = 2
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000, index_from = index_from)

words_to_search_for = ["ok", "just", "even", "or", "if", "they", "so"]


vocab = imdb.get_word_index()


word_to_id = imdb.get_word_index()
word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
word_to_id["<UNK>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}




counts_1 = [0, 0, 0, 0, 0, 0, 0]
counts_2 = [0, 0, 0, 0, 0, 0, 0]
lens_1 = 0
lens_2 = 0

for x in range(len(x_train)):
    do_print = False
    ids_to_sharpen = []
    for id in range(len(x_train[x])):
        for w in range(len(words_to_search_for)):
            if id_to_word[x_train[x][id]] == words_to_search_for[w]:
                if y_train[x] == 1:
                    counts_1[w] += 1
                else:
                    counts_2[w] += 1

                do_print = True
                ids_to_sharpen.append(id)
    if do_print:
        for id in range(len(x_train[x])):
            success = False
            for idd in ids_to_sharpen:
                if id == idd:
                    print("[" + id_to_word[x_train[x][id]] + "]", end=' ')
                    success = True
                    break
            if not success:
                print(id_to_word[x_train[x][id]], end=' ')

    print("")

diff_dict = {}
totals_dict = {}
p_dict = {}

n_dict = {}


for w in range(len(words_to_search_for)):
    diff_dict[words_to_search_for[w]] = (counts_2[w] - counts_1[w])
    n_dict[words_to_search_for[w]] = (counts_2[w])
    p_dict[words_to_search_for[w]] = (counts_1[w])
    totals_dict[words_to_search_for[w]] = (counts_2[w] + counts_1[w])

print("Negatives:", n_dict)
print("Positives:", p_dict)
print("Differences:", diff_dict)
print("Totals:", totals_dict)