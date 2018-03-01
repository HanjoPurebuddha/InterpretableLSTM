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
from keras.models import Model
from keras.layers import Input
from keras.datasets import imdb
from keras import backend as K
import keras
from itertools import product
import numpy as np
import random
import os
import ast
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

np.set_printoptions(suppress=True)

i_output_name = "PCAppmi0None20kCV1S0 SFT0 allL0305000LRkappa KMeans CA200 MC1 MS0.4 ATS2001 DS400 tdev"
n_clusters_fn = "PCAppmi0None20kCV1S0 SFT0 allL030LR5000kappa KMeans CA200 MC1 MS0.4 ATS2001 DS400"

save_format = "%1.3f"

max_features_a = [5000] # Was 20,000 S
maxlen_a = [300]  # cut texts after this number of words (among top max_features most common words) # L
batch_size_a = [25] # M
epochs_a = [64] #15,30,10 # L
dropout_a = [0.3] # L
recurrent_dropout_a = [0.05] # S
embedding_size_a = [16] # S
lstm_size_a = [32] # S
learn_rate_a = [0.001] # S

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

max_index_a = np.zeros(len(all_params), dtype="int")

scale_amount = 1
scale_amount_2 = 100
stateful =False
forget_bias = True
dev = True
use_all = False
iLSTM = True
iLSTM_t_model = False
rewrite = False
use_lr = False

def import1dArray(file_name, file_type="s"):
    with open(file_name, "r") as infile:
        if file_type == "f":
            array = []
            lines = infile.readlines()
            for line in lines:
                array.append(float(line.strip()))
        elif file_type == "i":
            array = [int(line.strip()) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return np.asarray(array)

import_model = "MF5000 ML300 BS25 FBTrue DO0.3 RDO0.05 E64 ES16LS200 UAFalse SFFalse iLTrue rTFalse lrFalse sA1"

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
    try:
        lstm_size = int(variables[7].split("L")[1][1:])
    except ValueError:
        lstm_size =  int(variables[7].split("L")[1][1:].split(".")[0])
        print(".Txt detected, split on '.', result is: " + str(lstm_size))
    try:
        stateful = ast.literal_eval(variables[9][2:])
    except IndexError:
        print("Stateful not included, set to default false value")
        stateful = False
    try:
        iLSTM = ast.literal_eval(variables[10][2:])
    except IndexError:
        print("iLSTM not included, set to default false value")
        iLSTM = False
    try:
        iLSTM_t_model = ast.literal_eval(variables[11][2:])
    except IndexError:
        print("iLSTM not included, set to default false value")
        iLSTM_t_model = False
    try:
        use_lr = bool(variables[12][2:])
    except IndexError:
        print("use_lr not included, set to default false value")
        use_lr = False
    try:
        scale_amount = float(variables[13][2:])
    except IndexError:
        print("scale_amt not included, set to default false value")
        scale_amount = 1




print('Loading data...')

# We need to load the IMDB dataset. We are constraining the dataset to the top 5,000 words. We also split the dataset into train (50%) and test (50%) sets.

data_path = "../data/sentiment/lstm/"


def saveVectors(m, v_fn):
    if os.path.exists(v_fn) is False:
        print("Output vectors")
        inp = m.input  # input placeholder
        outputs = [layer.output for layer in m.layers]  # all layer outputs
        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        # Note: To simulate Dropout use learning_phase as 1. in layer_outs otherwise use 0.
        layer_outs = functor([x_train, 1.])
        np.save(v_fn, layer_outs[1])
        print("Saved")
    else:
        print("Vectors already saved")


def saveModel(m, m_fn):
    if os.path.exists(m_fn) is False:
        print("Model")
        model.save(m_fn)
        print("Saved")
    else:
        print("Model already saved")



def saveScores(m, s_fn, development, x, y, rewrite, y_cell=None, y_names=None):
    a_fn = s_fn + " acc" + " d" + str(development)
    f1_fn = s_fn + " f1" + " d" + str(development)
    sc_fn = s_fn + " score" + " d" + str(development)
    if os.path.exists(sc_fn ) is False or os.path.exists(f1_fn) is False or rewrite:
        print("Scores")
        if y_cell is None:
            score, acc = m.evaluate(x, y,
                                        batch_size=batch_size)
            y_pred = m.predict_classes(x)
            f1 = f1_score(y, y_pred, average="binary")
        else:
            acc = m.evaluate(x, [y, y_cell], batch_size=batch_size)
            print("all returned",  acc)
            score = acc[2]
            ys_pred = m.predict(x, verbose=1, batch_size=batch_size)
            y_pred_a = np.asarray(ys_pred[0])
            y_pred = np.zeros(len(y_pred_a), dtype=np.float64)
            y_cell_pred = np.asarray(ys_pred[1])
            for i in range(len(y_pred)):
                y_pred[i] = y_pred_a[i][0]
            y_pred_classes = np.where(y_pred > 0.5, 1, y_pred)
            y_pred_classes = np.where(y_pred_classes <= 0.5, 0, y_pred_classes)
            y_pred_classes = y_pred_classes.astype(np.int64)
            f1 = f1_score(y, y_pred_classes, average="binary")
            compared_acc = accuracy_score(y, y_pred_classes)
            y_cell_pred = y_cell_pred.transpose()
            y_cell_pred_classes = np.empty(len(y_cell_pred), dtype=np.object)
            for c in range(len(y_cell_pred)):
                y_cell_pred_classes[c] = np.where(y_cell_pred[c] > 0.5, 1, y_cell_pred[c])
                y_cell_pred_classes[c] = np.where(y_cell_pred_classes[c] <= 0.5, 0, y_cell_pred_classes[c])
                y_cell_pred_classes[c] = y_cell_pred_classes[c].astype(np.int64)
            y_cell = y_cell.transpose()
            accs = np.zeros(len(y_cell_pred_classes), dtype=np.float64)
            f1s = np.zeros(len(y_cell_pred_classes), dtype=np.float64)
            nonzeros = np.zeros(len(y_cell), dtype=np.int64)
            for c in range(len(y_cell_pred_classes)):
                nonzeros[c] = np.count_nonzero(y_cell[c])
                cell_acc = accuracy_score(y_cell[c], y_cell_pred_classes[c])
                cell_f1 = f1_score(y_cell[c], y_cell_pred_classes[c], average="binary")
                print(y_names[c], "acc", cell_acc, "f1", cell_f1)
                accs[c] = cell_acc
                f1s[c] = cell_f1
            ids = np.flipud(np.argsort(f1s))
            f1s = f1s[ids]
            accs = accs[ids]
            top_clusters = y_names[ids]
            nonzeros = nonzeros[ids]
            for i in range(len(top_clusters)):
                print(nonzeros[i], "f1", f1s[i], "acc", accs[i], top_clusters[i])
            overall_acc = np.average(accs) # Macro accuracy score
            overall_f1 = np.average(f1s)
            print("overall y_cell acc", overall_acc)
            print("overall y_cell f1", overall_f1)

            acc = compared_acc
            f1 = f1

            #score_cell = score[1]
            #score = score[0]
            #y_pred = m.predict_classes([y, y_cell])
            #f1 = f1_score(y, y_pred[0])
            #f1_cell = f1_score(y_cell, y_pred[1])
            np.savetxt(f1_fn + " cell", [overall_f1], fmt=save_format)
            np.savetxt(a_fn + " cell", [overall_acc], fmt=save_format)
            np.savetxt(f1_fn + " cell_a", [f1s], fmt=save_format)
            np.savetxt(a_fn + " cell_a", [accs], fmt=save_format)

        print('Test score:', score)
        print('Test accuracy:', acc)
        print('Test f1:', f1)
        np.savetxt(a_fn, [acc], fmt=save_format)
        np.savetxt(sc_fn, [score], fmt=save_format)
        np.savetxt(f1_fn, [f1], fmt=save_format)
        print("Saved")
    else:
        acc = np.loadtxt(a_fn, dtype="float")
        score = np.loadtxt(sc_fn, dtype="float")
        f1 = np.loadtxt(a_fn, dtype="float")
        print("Scores already saved")
    return acc, score, f1


def saveState(m, s_fn, f_s_fn):
    if os.path.exists(f_s_fn + ".npy") is False:
        print("States")
        target_layer = model.layers[-2]
        target_layer.return_sequences = True
        outputs = target_layer(target_layer.input)
        m = keras.Model(model.input, outputs)
        hidden_states = m.predict(x_train)
        # np.save(s_fn, hidden_states)
        final_state = np.empty(shape=(len(x_train), lstm_size), dtype="object")
        for i in range(len(hidden_states)):
            final_state[i] = hidden_states[i][maxlen - 1]
        np.save(f_s_fn, final_state)
    else:
        print("States already saved")
    print("Saved")

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
    print(max_features, maxlen, batch_size, epochs, dropout, recurrent_dropout, embedding_size, lstm_size, stateful, iLSTM)
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
        else:
            j = len(all_params[i])


        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
        if iLSTM:
            y_cell = np.load("../data/sentiment/lstm/b_clusters/"+ i_output_name + ".npy").transpose()
            y_cell_train = y_cell[:len(x_train)]
            y_cell_test = y_cell[len(x_train):len(x_train) + len(x_test)]
        else:
            y_cell = None
            y_cell_train = None
            y_cell_test = None

        print("development data for hyperparameter tuning")


        if dev and not use_all:
            x_dev = x_train[int(len(x_train) * 0.8):]
            y_dev = y_train[int(len(y_train) * 0.8):]
            x_train = x_train[:int(len(x_train) * 0.8)]
            y_train = y_train[:int(len(y_train) * 0.8)]
            y_test = y_dev
            x_test = x_dev
            if iLSTM:
                y_cell_dev = y_cell_train[int(len(y_cell_train) * 0.8):]
                y_cell_train = y_cell_train[:int(len(y_cell_train) * 0.8)]
                y_cell_test = y_cell_dev
        elif use_all:
            x_train = np.concatenate((x_train, x_test))
            y_train = np.concatenate((y_train, y_test))
            x_test = x_train
            y_test = y_train
            if iLSTM:
                y_cell_train = np.concatenate((y_cell_train, y_cell_test))
                y_cell_test = y_cell_train
        elif not dev and not use_all:
            x_train = x_train[:int(len(x_train) * 0.8)]
            y_train = y_train[:int(len(y_train) * 0.8)]
            if iLSTM:
                y_cell_train = y_cell_train[:int(len(y_cell_train) * 0.8)]

        if iLSTM:
            lstm_size = len(y_cell_train[0])

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        """
        if iLSTM and not iLSTM_t_model:
            if use_lr:
                i_output = np.load("../data/sentiment/lstm/dp/" + i_output_name + ".npy")
            else:
                i_output = np.load("../data/sentiment/lstm/lr/" + i_output_name + ".npy")
                y_train = i_output.transpose()
                y_test = i_output.transpose()
            s_x_test = x_test
            s_y_train = y_train
            s_y_test = y_test
            s_x_train = x_train
            x_test = x_train
            #hs = np.load("../data/sentiment/lstm/states/" + "MF5000 ML300 BS25 FBTrue DO0.3 RDO0.05 E64 ES16LS32 UAFalse SFFalse iLTrue FState.npy")
        """
        # x_dev = x_train[:int(len(x_train)*0.2)]
        # y_dev = y_train[:int(len(y_train)*0.2)]
        # x_train = x_train[int(len(x_train)*0.2):]
        # y_train = y_train[int(len(y_train)*0.2):]

        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print(len(y_train), 'ytrain sequences')
        print(len(y_test), 'ytest sequences')
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        if iLSTM:
            print('y_cell_test shape:', y_cell_test.shape)
            print('y_cell_train shape:', y_cell_train.shape)
            print(len(y_cell_test), 'y_cell_test sequences')
            print(len(y_cell_train), 'y_cell_train sequences')

        # print(len(x_dev), 'dev train sequences')
        # print(len(y_dev), 'dev test sequences')

        # Next, we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras.
        """"""

        print('Build model...')
        if not import_model:
            file_name = "MF" + str(max_features) + " ML" + str(maxlen) + " BS" + str(batch_size) + " FB" + str(
            forget_bias) + " DO" \
                    + str(dropout) + " RDO" + str(recurrent_dropout) + " E" + str(epochs) + " ES" + str(
            embedding_size) + "LS" + \
                    str(lstm_size) + " UA" + str(use_all) + " SF" + str(stateful) + " iL" + str(iLSTM) + " rT" + \
                        str(iLSTM_t_model) + " lr" + str(use_lr) + " sA" + str(scale_amount)+ " sA2" + str(scale_amount_2)
        else:
            file_name = import_model
        print(file_name)

        vector_fn = data_path + "vectors/" + file_name + " L" + str(1)
        model_fn = data_path + "model/" + file_name
        score_fn = data_path + "score/" + file_name
        state_fn = data_path + "states/" + file_name + " HState"
        final_state_fn = data_path + "states/" + file_name + " FState"

        input_layer = None
        embedding_layer = None
        cell_state = None
        output_layer = None
        model = None

        if iLSTM:
            input_layer = Input(shape=(maxlen,))  #
            embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size)(input_layer)
            h_l1, h_l2, cell_state = LSTM(units=lstm_size, recurrent_activation="hard_sigmoid",
                                          unit_forget_bias=forget_bias, activation="tanh",
                                          dropout=dropout, recurrent_dropout=recurrent_dropout,
                                          kernel_initializer="glorot_uniform", stateful=stateful, return_state=True)(
                embedding_layer)
            output_layer = Dense(1, activation='sigmoid')(h_l1)
            model = Model(input_layer, [output_layer, cell_state])

        if import_model is None and (os.path.exists(vector_fn) is False or os.path.exists(model_fn) is False or os.path.exists(score_fn) is False or os.path.exists(state_fn) is False):

            if iLSTM:
                model.compile(loss=['binary_crossentropy', 'mse'],
                              optimizer='adam',
                              metrics=['accuracy'], loss_weights=[1.0*scale_amount,1.0])
                print('Train...')

                model.fit(x_train, [y_train, y_cell_train],
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, [y_test, y_cell_test]))
            else:
                if stateful:
                    print("STATEFUL")
                    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size, batch_input_shape=(batch_size, maxlen))
                else:
                    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size)
                hidden_layer = LSTM(units=lstm_size, recurrent_activation="hard_sigmoid", unit_forget_bias=forget_bias, activation="tanh",
                     dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_initializer="glorot_uniform", stateful=stateful)
                output_layer = Dense(1, activation='sigmoid')
                model = Model(embedding_layer, output_layer)
                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                print('Train...')
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

            # Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is          fit for only 2 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.

            # try using different optimizers and different optimizer configs
        elif import_model is not None:
            print("Loading model...")
            model = keras.models.load_model("../data/sentiment/lstm/model/" + import_model )
        saveModel(model, model_fn)
        if iLSTM:
            y_names = import1dArray("../data/sentiment/lstm/n_clusters/"+n_clusters_fn+".txt", "s")
        else:
            y_names = None

        acc, score, f1 = saveScores(model, score_fn, dev, x_test, y_test, rewrite, y_cell=y_cell_test, y_names=y_names)
        if not iLSTM:
            saveVectors(model, vector_fn)
            saveState(model, state_fn, final_state_fn)
        # Do the thing to get the acc
        if acc > max_acc:
            max_index = j
            max_acc = acc

        # Train the new supervised model
        if iLSTM and not iLSTM_t_model:
            file_name = "MF" + str(max_features) + " ML" + str(maxlen) + " BS" + str(batch_size) + " FB" + str(
                forget_bias) + " DO" \
                        + str(dropout) + " RDO" + str(recurrent_dropout) + " E" + str(epochs) + " ES" + str(
                embedding_size) + "LS" + \
                        str(lstm_size) + " UA" + str(use_all) + " SF" + str(stateful) + " iL" + str(iLSTM) + " rTTrue" + \
                        " lr" + str(use_lr)+ " sA" + str(scale_amount)+ " sA2" + str(scale_amount_2)

            print(file_name)

            vector_fn = data_path + "vectors/" + file_name + " L" + str(1)
            model_fn = data_path + "model/" + file_name
            acc_fn = data_path + "score/" + file_name + " acc"
            score_fn = data_path + "score/" + file_name + " score"
            state_fn = data_path + "states/" + file_name + " HState"
            final_state_fn = data_path + "states/" + file_name + " FState"

            model.compile(loss=['binary_crossentropy', 'mse'],
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[1.0 * scale_amount_2, 1.0])
            print('Train...')

            model.fit(x_train, [y_train, y_cell_train],
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, [y_test, y_cell_test]))

            print(len(x_train), 'train sequences')
            print(len(x_test), 'test sequences')
            print(len(y_train), 'ytrain sequences')
            print(len(y_test), 'ytest sequences')
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)
            print('y_train shape:', y_train.shape)
            print('y_test shape:', y_test.shape)

            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test))

            saveModel(model, model_fn)
            acc, score, f1 = saveScores(model, score_fn, dev, x_test, y_test, y_cell=y_cell_test, y_names=y_names)

            if acc > max_acc:
                max_index = j
                max_acc = acc




        if import_model is not None:
            print("Complete!")
            break

    if import_model is not None:
        print("Complete!")
        break
    if import_model is None:
        print("max option for param", i, "is", max_index, "with", max_acc)
        max_index_a[i] = max_index
        np.savetxt(data_path + "score/" + "max vals" + str(i) + " acc", max_index_a, fmt=save_format)
        np.savetxt(data_path + "score/" + "all params" + str(i) +  " acc", all_params, fmt=save_format)

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




