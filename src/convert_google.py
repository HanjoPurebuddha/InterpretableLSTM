import os
import read_text as rt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
words_to_match = rt.importArray("../data/sentiment/wordvectors/words/0-None-all.txt", encoding="cp1252")

embeddings_index = {}
wv = KeyedVectors.load_word2vec_format("../data/sentiment/wordvectors/data/google/GoogleNews-vectors-negative300.bin", binary=True)
wv.get_vector(words_to_match[w])
entities = wv.index2entity

embedding_matrix = np.zeros((len(words_to_match) + 1, 300))
ind_to_delete = []
for w in range(len(words_to_match)):
    try:
        embedding_vector = wv.get_vector(words_to_match[w])
        embedding_matrix[w] = embedding_vector
    except KeyError:
        print(KeyError)
        ind_to_delete.append(w)

words_to_match = np.delete(words_to_match, ind_to_delete)
file_name = "google_news_no_bigram_IMDB_words"

rt.writeArray(words_to_match, "../data/sentiment/wordvectors/words/" + file_name + ".txt")
np.save("../data/sentiment/wordvectors/vectors/" + file_name + ".npy", embedding_matrix)
