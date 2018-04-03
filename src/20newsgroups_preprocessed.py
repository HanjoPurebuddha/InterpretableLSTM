'''
ORIGINAL COMMENT:

This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html

ADDENDUM:
The 20 newsgroups data above contains duplicates, so it does not follow
standard splits for the data. Below, we use the scikit-learn method to
obtain the data instead.
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
from gensim.models.keyedvectors import KeyedVectors

GLOVE_DIR = os.path.join("/home/tom/Downloads/", 'glove.6B')
TEXT_DATA_DIR = os.path.join("/home/tom/Downloads/", '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')


all_vectors = KeyedVectors.load_word2vec_format(
    "../data/sentiment/wordvectors/data/google/GoogleNews-vectors-negative300.bin", binary=True)



print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

# Remove headers, footers and quotes so as to not overfit on metadata
all = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))


print("Concatenating")

texts = all.data
labels = all.target

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Print an example to check everything is working
example_to_print = 5
word_index = tokenizer.word_index
word_to_id = word_index
word_to_id = {k:(v) for k,v in word_to_id.items()}
id_to_word = {value:key for key,value in word_to_id.items()}
for id in sequences[example_to_print]:
    print(id, end=' ')
print("")
for id in sequences[example_to_print]:
    print(id_to_word[id], end=' ')


print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # left-padding of zeros on sequences that aren't as long as the max

labels = to_categorical(np.asarray(labels)) # convert single list of [0,1,2,3,...,20] labels to 20 [0,0,1,...,0,0,1] binary-labels
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split into specific splits

x_train = data[:11314]
x_test = data[11314:]
y_train = labels[:11314]
y_test = labels[11314:]

# Add development set

x_dev = x_train[int(len(x_train) * 0.8):]
y_dev = y_train[int(len(y_train) * 0.8):]
x_train = x_train[:int(len(x_train) * 0.8)]
y_train = y_train[:int(len(y_train) * 0.8)]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


"""
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(y_train[0]), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_dev, y_dev))

print(model.evaluate(x_dev, y_dev))
"""