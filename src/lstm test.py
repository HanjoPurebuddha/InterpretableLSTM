'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

#CD PAPER PARAMETERS:
#We implemented all models in Torch using default hyperparameters for weight initializa- tions. All models were optimized using Adam (Kingma & Ba, 2014) with the default learning rate of 0.001 using early stopping on the validation set. For the linear model, we used a bag of vectors model, where we sum pre-trained Glove vectors (Pennington et al., 2014) and add an additional lin- ear layer from the word embedding dimension, 300, to the number of classes, 2. We fine tuned both the word vectors and linear parameters.


# LSTM Parameters
# units: Positive integer, dimensionality of the output space.
#  using a units number multiple of 32 may actually speed up a little the training (When you dealing with float32). So usually, people will do as such when designing their RNN.
# Units = the amount of dimensions of the output space
# activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear" activation:  a(x) = x). Default is tanh.
# The activation function for updating the cell-state and when outputting
# recurrent_activation: Activation function to use for the recurrent step (see activations) Default is hard_sigmoid.
# The gating function for the three gates.
# unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.
# Bias on the forget gate as "The rationale is that the network shouldn't forget past information until it has learnt to forget it, and that it shouldn't bring in new info unless it has learnt that it is good to add new info to its internal state. While that seems sensible I have little idea whether it is useful in practice."
# No regularizer or constraints used by default
# dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
# recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
# return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
# return_state: Boolean. Whether to return the last state in addition to the output.
# go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
# stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
# unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.


from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras import backend as K
import keras
from itertools import product
import numpy as np
import random
import os

max_features_a = [5000] # Was 20,000 S
maxlen_a = [300]  # cut texts after this number of words (among top max_features most common words) # L
batch_size_a = [25] # M
epochs_a = [64] #15,30,10 # L
dropout_a = [0.3] # L
recurrent_dropout_a = [0.05] # S
embedding_size_a = [16] # S
lstm_size_a = [32] # S
learn_rate_a = [0.001] # S
stateful = [False]

all_params = []

all_params.append(max_features_a)
all_params.append(maxlen_a)
all_params.append(batch_size_a)
all_params.append(epochs_a)
all_params.append(dropout_a)
all_params.append(recurrent_dropout_a)
all_params.append(embedding_size_a)
all_params.append(lstm_size_a)
all_params.append(learn_rate_a)
all_params.append(stateful)

max_index_a = np.zeros(len(all_params), dtype="int")

forget_bias = True
dev = True
use_all = False
iLSTM = True
dp_fn = "LSTMFstate5k30032CV1S0 SFT0 allL030LRkappa KMeans CA64 MC1 MS0.4 ATS1000 DS128 tdev"


import_model = None#"MF5000 ML300 BS32 FBTrue DO0.3 RDO0.05 E64 ES16LS32"

if import_model is not None:
    variables = import_model.split()
    max_features = int(variables[0][2:])
    maxlen = int(variables[1][2:])
    batch_size = int(variables[2][2:])
    epochs = int(variables[6][1:])
    forget_bias = bool(variables[3][2:])
    dropout = float(variables[4][2:])
    recurrent_dropout = float(variables[5][3:])
    embedding_size = int(variables[7].split("L")[0][2:])
    lstm_size = int(variables[7].split("L")[1][1:])
    try:
        stateful = bool(variables[9][2:])
    except IndexError:
        print("Stateful not included, set to default false value")
        stateful = False
    try:
        iLSTM = bool(variables[10][2:])
    except IndexError:
        print("iLSTM not included, set to default false value")
        stateful = False

print('Loading data...')

# We need to load the IMDB dataset. We are constraining the dataset to the top 5,000 words. We also split the dataset into train (50%) and test (50%) sets.

data_path = "../data/sentiment/lstm/"

for i in range(len(all_params)):
    if import_model is None:
        max_features = all_params[0][max_index_a[0]]
        maxlen = all_params[1][max_index_a[1]]
        batch_size = all_params[2][max_index_a[2]]
        epochs = all_params[3][max_index_a[3]]
        dropout = all_params[4][max_index_a[4]]
        recurrent_dropout = all_params[5][max_index_a[5]]
        embedding_size = all_params[6][max_index_a[6]]
        lstm_size = all_params[7][max_index_a[7]]
        stateful = all_params[8][max_index_a[8]]
    print(max_features, maxlen, batch_size, epochs, dropout, recurrent_dropout, embedding_size, lstm_size)
    max_index = 0
    max_acc = 0
    for j in range(len(all_params[i])):
        if import_model is None:
            if i == 0:
                max_features = all_params[0][max_index_a[j]]
            if i == 1:
                maxlen = all_params[1][max_index_a[j]]
            if i == 2:
                batch_size = all_params[2][max_index_a[j]]
            if i == 3:
                epochs = all_params[3][max_index_a[j]]
            if i == 4:
                dropout = all_params[4][max_index_a[j]]
            if i == 5:
                recurrent_dropout = all_params[5][max_index_a[j]]
            if i == 6:
                embedding_size = all_params[6][max_index_a[j]]
            if i == 7:
                lstm_size = all_params[7][max_index_a[j]]
            if i == 8:
                stateful = all_params[8][stateful[j]]
        else:
            j = len(all_params[i])

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

        print("development data for hyperparameter tuning")



        if dev and not use_all:
            x_dev = x_train[int(len(x_train) * 0.8):]
            y_dev = y_train[int(len(y_train) * 0.8):]
            x_train = x_train[:int(len(x_train) * 0.8)]
            y_train = y_train[:int(len(y_train) * 0.8)]
            y_test = y_dev
            x_test = x_dev
        elif use_all:
            x_train = np.concatenate((x_train, x_test))
            y_train = np.concatenate((y_train, y_test))
            x_test = x_train
            y_test = y_train

        if iLSTM:
            dp = np.load("../data/sentiment/lstm/dp/" + dp_fn + ".npy")
            y_train = dp.transpose()
            y_test = dp.transpose()
            x_test = x_train

        # x_dev = x_train[:int(len(x_train)*0.2)]
        # y_dev = y_train[:int(len(y_train)*0.2)]
        # x_train = x_train[int(len(x_train)*0.2):]
        # y_train = y_train[int(len(y_train)*0.2):]

        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        # print(len(x_dev), 'dev train sequences')
        # print(len(y_dev), 'dev test sequences')

        # Next, we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras.
        """"""
        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)


        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        print('Build model...')
        file_name = "MF" + str(max_features) + " ML" + str(maxlen) + " BS" + str(batch_size) + " FB" + str(
            forget_bias) + " DO" \
                    + str(dropout) + " RDO" + str(recurrent_dropout) + " E" + str(epochs) + " ES" + str(
            embedding_size) + "LS" + \
                    str(lstm_size) + " UA" + str(use_all) + "SF" + str(stateful) + " iL" + str(iLSTM)

        vector_fn = data_path + "vectors/" + file_name + " L" + str(1)
        model_fn = data_path + "model/" + file_name
        score_fn = data_path + "score/" + file_name + " acc"
        state_fn = data_path + "states/" + file_name + " HState"
        final_state_fn = data_path + "states/" + file_name + " FState"

        if import_model is None and (os.path.exists(vector_fn) is False or os.path.exists(model_fn) is False or os.path.exists(score_fn) is False or os.path.exists(state_fn) is False):
            model = Sequential()

            # The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
            if stateful:
                model.add(Embedding(input_dim=max_features, output_dim=embedding_size, batch_input_shape=(batch_size, maxlen)))
            else:
                model.add(Embedding(input_dim=max_features, output_dim=embedding_size))

            # Input = size of vocab, output = dimension of dense embedding

            if not iLSTM:
                model.add(
                LSTM(units=lstm_size, recurrent_activation="hard_sigmoid", unit_forget_bias=forget_bias, activation="tanh",
                     dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_initializer="glorot_uniform", stateful=stateful))
            else:
                model.add(
                LSTM(units=lstm_size, recurrent_activation="hard_sigmoid", unit_forget_bias=forget_bias, activation="linear",
                     dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_initializer="glorot_uniform", stateful=stateful))

            if not iLSTM:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(len(y_train[0]), activation='linear'))


            # Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is          fit for only 2 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.

            # try using different optimizers and different optimizer configs

            if not iLSTM:
                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
            else:
                model.compile(loss='mse',
                              optimizer='adam',
                              metrics=['accuracy'])


            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test))
        elif import_model is not None:
            model = keras.models.load_model("../data/sentiment/lstm/model/" + import_model + ".txt")

        if os.path.exists(vector_fn) is False:
            if stateful is False:
                print("Output vectors")
                inp = model.input  # input placeholder
                outputs = [layer.output for layer in model.layers]  # all layer outputs
                functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
                # Note: To simulate Dropout use learning_phase as 1. in layer_outs otherwise use 0.
                layer_outs = functor([x_train, 1.])
                np.save(vector_fn, layer_outs[1])

                print("Saved")



        if os.path.exists(model_fn) is False:
            print("Model")
            model.save(data_path + "model/" + file_name)
            print("Saved")


        if os.path.exists(score_fn) is False:
            print("Scores")
            score, acc = model.evaluate(x_test, y_test,
                                        batch_size=batch_size)
            print('Test score:', score)
            print('Test accuracy:', acc)


            np.savetxt(data_path + "score/" + file_name + " acc", [acc])
            np.savetxt(data_path + "score/" + file_name + " score", [score])
            print("Saved")

            # Do the thing to get the acc
            if acc > max_acc:
                max_index = j
                max_acc = acc

        if os.path.exists(state_fn) is False:
            print("States")
            target_layer = model.layers[-2]
            target_layer.return_sequences = True
            outputs = target_layer(target_layer.input)
            m = keras.Model(model.input, outputs)
            hidden_states = m.predict(x_train)
            np.save(state_fn, hidden_states)
            final_state = np.empty(shape=(len(x_train),lstm_size), dtype="object")
            for i in range(len(hidden_states)):
                final_state[i] = hidden_states[i][maxlen-1]
            np.save(final_state_fn, final_state)
            """
            inp = model.input  # input placeholder
            outputs = [layer.output for layer in model.layers]  # all layer outputs
            functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
            # Note: To simulate Dropout use learning_phase as 1. in layer_outs otherwise use 0.
            layer_outs = functor([x_train, 1.])

            #np.save(data_path + "states/" + file_name + " C", states[0])
            """
            print("Saved")


        if import_model is not None:
            print("Complete!")
            break

    if import_model is not None:
        print("Complete!")
        break
    if import_model is None:
        print("max option for param", i, "is", max_index, "with", max_acc)
        max_index_a[i] = max_index
        np.savetxt(data_path + "score/" + "max vals" + str(i) + " acc", max_index_a)
        np.savetxt(data_path + "score/" + "all params" + str(i) +  " acc", all_params)

# Generate parameter list
params = []
"""
for max_features in max_features_a:
    for maxlen in maxlen_a:
        for batch_size in batch_size_a:
            for forget_bias in forget_bias_a:
                for epochs in epochs_a:
                    for dropout in dropout_a:
                        for recurrent_dropout in recurrent_dropout_a:
                            for embedding_size in embedding_size_a:
                                for lstm_size in lstm_size_a:
"""




