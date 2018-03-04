from gensim.models.word2vec import Word2Vec
import numpy as np
import read_text as rt
from gensim.models import Phrases


file_structure = "../data/sentiment/wordvectors/data/"

# Import all data
compiled_names = ["test_pos", "test_neg", "train_pos", "train_neg", "unsup"]
sentences = []
for i in range(len(compiled_names)):
    array = rt.importArrays(file_structure + "compiled/" + compiled_names[i] + ".txt")
    for s in range(len(array)):
        sentences.append(array[s])

sentences = np.asarray(sentences)
np.random.shuffle(sentences)

size = 100 # For sentiment, larger values work better
window = 5 # Standard window size
min_count = 5 # Standard min_count
sg = 1 # Set to skip gram, works better for smaller datasets
phrases = True

file_name = "size" + str(size) + " window" + str(window) + " min_count" + str(min_count) + " sg" + str(sg) + " phrases" + str(phrases)

if phrases: # Convert to bigrams, can be iteratively applied e.g. trigrams = Phrases(bigrams[sentences])
    bigrams = Phrases(sentences)
    sentences = bigrams[sentences]



# Train word2vec model and save it
model = Word2Vec(sentences, sg=sg, size=size, window=window, min_count=min_count, workers=3, sorted_vocab=1)

model.save("../data/sentiment/wordvectors/models/" + file_name +".model")
model.wv.save( "../data/sentiment/wordvectors/wv/" + file_name + ".wv")

# Get the word representations
entities = model.wv.index2entity
vectors = []
for e in range(len(entities)):
    vector = model.wv.get_vector(entities[e])
    vectors.append(vector)
    print(e)

rt.writeArray(entities,  "../data/sentiment/wordvectors/words/" + file_name + ".txt")
np.save( "../data/sentiment/wordvectors/vectors/" + file_name + ".npy", vectors)

