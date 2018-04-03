
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os
from keras import backend as K
import numpy as np
import keras
import read_text as rt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Input, CuDNNLSTM, Dropout, Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K
import keras
from keras.optimizers import Adam
from itertools import product
import parse_binary_treebank as ptb
from sklearn.datasets import fetch_20newsgroups
import process_20newsgroups as p20

save_format = "%1.3f"

def getData(max_features, iLSTM, i_output_name, dev, use_all, test, maxlen, dataset, data_path):
    if dataset == 0:
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    elif dataset == 1:
        x_train, x_test, x_dev, y_train, y_test, y_dev = ptb.loadProcessedSplits()
    elif dataset == 2:
        x_train, x_test,  y_train, y_test = p20.loadProcessedSplits()
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print(len(y_train), 'ytrain sequences')
    print(len(y_test), 'ytest sequences')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    if iLSTM:

        y_cell = np.load(data_path + "b_clusters/" + i_output_name + ".npy").transpose()
        y_cell_train = y_cell[:len(x_train)]
        y_cell_test = y_cell[len(x_train):len(x_train) + len(x_test)]
    else:
        y_cell = None
        y_cell_train = None
        y_cell_test = None

    print("development data for hyperparameter tuning")

    if dev and not use_all:
        if dataset != 1:
            x_dev = x_train[int(len(x_train) * 0.8):]
            y_dev = y_train[int(len(y_train) * 0.8):]
            x_train = x_train[:int(len(x_train) * 0.8)]
            y_train = y_train[:int(len(y_train) * 0.8)]
            if iLSTM:
                y_cell_dev = y_cell_train[int(len(y_cell_train) * 0.8):]
                y_cell_train = y_cell_train[:int(len(y_cell_train) * 0.8)]
                y_cell_test = y_cell_dev
        else:
            if iLSTM:
                y_cell_dev = y_cell_test[:872]
                y_cell_test = y_cell_test[872:]
                y_cell_test = y_cell_dev
        y_test = y_dev
        x_test = x_dev
    elif use_all:
        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))
        x_test = x_train
        y_test = y_train
        if iLSTM:
            y_cell_train = np.concatenate((y_cell_train, y_cell_test))
            y_cell_test = y_cell_train
    elif not dev and not use_all:
        if dataset != 1:
            x_train = x_train[:int(len(x_train) * 0.8)]
            y_train = y_train[:int(len(y_train) * 0.8)]
            if iLSTM:
                y_cell_train = y_cell_train[:int(len(y_cell_train) * 0.8)]

    if iLSTM:
        lstm_size = len(y_cell_train[0])

    if test:
        x_train = x_train[:100]
        y_train = y_train[:100]
        x_test = x_test[:100]
        y_test = y_test[:100]
        y_cell_train = y_cell_train[:100]
        y_cell_test = y_cell_test[:100]

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

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
    return x_train, y_train, x_test, y_test, imdb.get_word_index(), y_cell_test, y_cell_train


def getEmbeddingMatrix(word_vector_fn, word_index, dataset, orig_fn):
    matrix_fn =  orig_fn + "matrix/" + word_vector_fn + ".npy"
    if os.path.exists(matrix_fn):
        embedding_matrix = np.load(matrix_fn)
    else:
        if dataset == 0:
            word_vectors = np.load(orig_fn + "wordvectors/vectors/" + word_vector_fn + ".npy")
            word_vector_entities = rt.importArray(orig_fn + "wordvectors/words/" + word_vector_fn + ".txt",
                                                  encoding="cp1252")
            word_dict = {}
            for w in range(len(word_vector_entities)):
                word_dict[word_vector_entities[w]] = word_vectors[w]
            embedding_matrix = np.zeros((len(word_index) + 1, len(word_vectors[0])))
            for word, w in word_index.items():
                embedding_vector = word_dict.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[w] = embedding_vector
            np.save(matrix_fn, embedding_matrix)
        elif dataset == 1 or dataset == 2:
            word_vectors = rt.load_dict(orig_fn + "raw/wv_vocab.dict")
            embedding_matrix = np.zeros((len(word_vectors.keys()), len(word_vectors[0]))) # movie is magic keyword. we assume it exists as word vector
            for key, item in word_vectors.items():
                embedding_matrix[key] = item
    return embedding_matrix, len(embedding_matrix[0]), len(embedding_matrix)

from keras.callbacks import TensorBoard
"""
def getModel(import_model, max_features, maxlen, embedding_size, lstm_size, forget_bias, recurrent_dropout,
             dropout, stateful, iLSTM, scale_amount, x_train, y_train, x_test, y_test, y_cell_train, y_cell_test, wv_size,
             wi_size, batch_size, epochs, embedding_matrix, model_fn, file_name, dataset, use_wv, data_path):
    print('Build model...')
    tensorboard = TensorBoard(log_dir='/home/tom/Desktop/Logs/'+str(dataset)+"/"+file_name+'/', histogram_freq=0,
                              write_graph=True, write_images=True)

    model = None

    if iLSTM:
        input_layer = Input(shape=(maxlen,))  #
        embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size)(input_layer)
        h_l1, h_l2, cell_state = CuDNNLSTM(units=lstm_size,
                                      unit_forget_bias=forget_bias, return_state=True)(
            embedding_layer)
        output_layer = Dense(1, activation='sigmoid')(h_l1)
        model = Model(input_layer, [output_layer, cell_state])

    if import_model is None and os.path.exists(model_fn) is False:

        if iLSTM:
            model.compile(loss=['binary_crossentropy', 'mse'],
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[1.0 , 1.0 / scale_amount])
            print('Train...')

            model.fit(x_train, [y_train, y_cell_train],
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, [y_test, y_cell_test]), callbacks=[tensorboard])
        else:
            sequence_input = Input(shape=(maxlen,), dtype='int32')

            embedding_layer = Embedding(wi_size, wv_size, weights=[embedding_matrix],
                                        input_length=maxlen, trainable=False)(sequence_input)

            hidden_layer = CuDNNLSTM(units=lstm_size, recurrent_activation="hard_sigmoid", unit_forget_bias=forget_bias)(embedding_layer)

            output_layer = Dense(1, activation='sigmoid')(hidden_layer)

            model = Model(sequence_input, output_layer)
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test), callbacks=[tensorboard])

            # Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is          fit for only 2 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.

            # try using different optimizers and different optimizer configs
    elif import_model is not None:
        print("Loading model...")
        model = keras.models.load_model("../data/sentiment/lstm/model/" + import_model)
    else:
        model = keras.models.load_model(model_fn)
    return model
"""
def getModel(import_model, max_features, maxlen, embedding_size, lstm_size, forget_bias, recurrent_dropout,
             dropout, stateful, iLSTM, scale_amount, x_train, y_train, x_test, y_test, y_cell_train, y_cell_test, wv_size,
             wi_size, batch_size, epochs, embedding_matrix, model_fn, file_name, dataset, use_wv, data_path, trainable):
    print('Build model...')

    if dataset == 2:
        output_size = len(y_train[0])
        output_activation = "softmax"
        loss = "categorical_crossentropy"
    else:
        output_size = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"

    tensorboard = TensorBoard(log_dir='/home/tom/Desktop/Logs/'+str(dataset)+"/"+file_name+'/', histogram_freq=0,
                              write_graph=True, write_images=True)

    if y_cell_train is not None:
        print("interpretable LSTM")
        lstm_size = len(y_cell_train[0])
    else:
        print("normal LSTM")
    model = None
    print("lstm size", lstm_size)

    if import_model is None and os.path.exists(model_fn) is False:
        print("L0 Input layer", maxlen)
        sequence_input = Input(shape=(maxlen,), dtype=np.int32)  #

        if use_wv:
            print("L1 pre-trained word embeddings", wi_size, wv_size, maxlen, False, trainable)
            embedding_layer = Embedding(wi_size, wv_size, weights=[embedding_matrix],
                                        input_length=maxlen, trainable=trainable)(sequence_input)
        else:
            print("L1 trainable embeddings", wi_size, wv_size, maxlen, True)
            embedding_layer = Embedding(wi_size, wv_size,
                                        input_length=maxlen, trainable=True)(sequence_input)
        if iLSTM:
            if dropout > 0.0 or recurrent_dropout > 0.0:
                print("L2 dropout LSTM", lstm_size, forget_bias, dropout, recurrent_dropout)
                hidden_layer, h_l2, cell_state = keras.layers.LSTM(units=lstm_size, dropout=dropout,
                                                                   recurrent_dropout=recurrent_dropout,
                                                                   unit_forget_bias=forget_bias, return_state=True)(
                    embedding_layer)
            else:
                print("L2 no_dropout CuDNNLSTM", lstm_size, forget_bias)
                hidden_layer, h_l2, cell_state = keras.layers.CuDNNLSTM(units=lstm_size, unit_forget_bias=forget_bias,
                                                                        return_state=True)(embedding_layer)
        else:
            if dropout > 0.0 or recurrent_dropout > 0.0:
                print("L2 dropout LSTM", lstm_size, forget_bias, dropout, recurrent_dropout)

                hidden_layer = keras.layers.LSTM(units=lstm_size, dropout=dropout,
                                                 recurrent_dropout=recurrent_dropout,
                                                 unit_forget_bias=forget_bias)(embedding_layer)
            else:
                print("L2 no_dropout CuDNNLSTM", lstm_size, forget_bias)
                hidden_layer = keras.layers.CuDNNLSTM(units=lstm_size, unit_forget_bias=forget_bias)(embedding_layer)

        print("L3 output layer", output_size, output_activation)
        output_layer = Dense(output_size, activation=output_activation)(hidden_layer)

        if iLSTM:
            model = Model(sequence_input, [output_layer, cell_state])
            model.compile(loss=[loss, 'mse'],
                          optimizer='adam',
                          metrics=['accuracy'], loss_weights=[1.0, 1.0 / scale_amount])
            print('Train...')
            history = model.fit(x_train, [y_train, y_cell_train],
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_test, [y_test, y_cell_test]), callbacks=[tensorboard])

        else:
            model = Model(sequence_input, output_layer)
            model.compile(loss=loss,
                          optimizer='adam',
                          metrics=['accuracy'])

            print('Train...')
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_test, y_test), callbacks=[tensorboard])
        print(history)
    elif import_model is not None:
        print("Loading model...")
        model = keras.models.load_model(data_path+"model/" + import_model)
    else:
        model = keras.models.load_model(model_fn)

    return model

def saveVectors(m, v_fn, x):
    print("removed due to embedding layer being too large")
    """
    if os.path.exists(v_fn) is False:
        print("Output vectors")
        inp = m.input  # input placeholder
        outputs = [layer.output for layer in m.layers]  # all layer outputs
        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        # Note: To simulate Dropout use learning_phase as 1. in layer_outs otherwise use 0.
        layer_outs = functor([x, 1.])
        np.save(v_fn, layer_outs[1])
        print("Saved")
    else:
        print("Vectors already saved")
    """


def saveModel(m, m_fn):
    if os.path.exists(m_fn) is False:
        print("Model")
        m.save(m_fn)
        print("Saved")
    else:
        print("Model already saved")



def saveScores(m, s_fn, development, x, y, rewrite, batch_size, y_cell=None, y_names=None, dataset=1):
    a_fn = s_fn + " acc" + " d" + str(development)
    f1_fn = s_fn + " f1" + " d" + str(development)
    sc_fn = s_fn + " score" + " d" + str(development)
    if os.path.exists(sc_fn ) is False or os.path.exists(f1_fn) is False or rewrite:
        print("Scores")
        if y_cell is not None:
            vals =  m.evaluate(x, [y, y_cell], batch_size=batch_size)
            score = vals[2]
        else:
            score, acc = m.evaluate(x, y, batch_size=batch_size)
        print("Score,", score)
        y_pred = m.predict(x, verbose=1, batch_size=batch_size)
        if y_cell is not None:
            y_pred_a = np.asarray(y_pred[0])
            y_cell_pred = np.asarray(y_pred[1])
            if len(y_pred_a) <= 1:
                y_pred = np.zeros(len(y_pred_a), dtype=np.float64)
                for i in range(len(y_pred)):
                    y_pred[i] = y_pred_a[i][0]
            else:
                y_pred = y_pred_a
        thresholds = np.arange(0.0, 1.0, 0.1)
        highest_score = 0.0
        threshold = 0.0
        for t in thresholds:
            y_pred_classes = np.where(y_pred > t, 1, y_pred)
            y_pred_classes = np.where(y_pred_classes <= t, 0, y_pred_classes)
            if dataset != 2:
                f1 = f1_score(y, y_pred_classes, average="binary")
            else:
                f1 = f1_score(y, y_pred_classes, average="macro")
            if f1 > highest_score:
                highest_score = f1
                threshold = t
            if t == 0.5:
                print("Baseline f1", f1)
            else:
                print("f1", f1)
        y_pred_classes = np.where(y_pred > threshold, 1, y_pred)
        y_pred_classes = np.where(y_pred_classes <= threshold, 0, y_pred_classes)
        acc = accuracy_score(y, y_pred_classes)
        if dataset != 2:
            f1 = f1_score(y, y_pred_classes, average="binary")
        else:
            f1 = f1_score(y, y_pred_classes, average="macro")
        print("Final f1", f1)
        print("Final acc", acc)
        if y_cell is not None:
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
            overall_acc = np.average(accs)  # Macro accuracy score
            overall_f1 = np.average(f1s)
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
    return acc, 0.0, f1


def saveState(m, s_fn, x, lstm_size, maxlen, f_s_fn, iLSTM):
    if os.path.exists(f_s_fn + ".npy") is False:
        print("States")
        target_layer = m.layers[-2]
        target_layer.return_state = True
        outputs = target_layer(target_layer.input)
        m = keras.Model(m.input, outputs)
        hidden_states = m.predict(x)
        final_state = hidden_states[2]
        # np.save(s_fn, hidden_states)
        np.save(f_s_fn, final_state)
    else:
        print("States already saved")
    print("Saved")
