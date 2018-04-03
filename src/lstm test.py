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
#  using a units number multiple of 32 may actually speed up a little the training (When you dealing with float32).
# So usually, people will do as such when designing their RNN.
# Units = the amount of dimensions of the output space
# activation: Activation function to use (see activations). If you pass None, no activation is applied (ie. "linear"
# activation:  a(x) = x). Default is tanh.
# The activation function for updating the cell-state and when outputting
# recurrent_activation: Activation function to use for the recurrent step (see activations) Default is hard_sigmoid.
# The gating function for the three gates.
# unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will
# also force  bias_initializer="zeros". This is recommended in Jozefowicz et al.
# Bias on the forget gate as "The rationale is that the network shouldn't forget past information until it has learnt to
# forget it, and that it shouldn't bring in new info unless it has learnt that it is good to add new info to its
# internal state. While that seems sensible I have little idea whether it is useful in practice."
# No regularizer or constraints used by default
# dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
# recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent
# state.
# return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
# return_state: Boolean. Whether to return the last state in addition to the output.
# go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
# stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as
# initial state for the sample of index i in the following batch.
# unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used.
# Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.


from __future__ import print_function


import numpy as np
import ast
import manage

np.random.seed(1337) # for reproducibility


dataset = 0

if dataset == 0:
    n_clusters_fn_a = ["wikiF300LSTMCstate5010kCV1S0 SFT0 allL0301000 LR acc KMeans CA50 MC1 MS0.4 ATS50 DS50",
                       "wiki300LSTMCstate5010kCV1S0 SFT0 allL0301000 LR acc wvhalf  KMeans CA50 MC1 MS0.4 ATS200 DS100",
                       "wiki300LSTMCstate5010kCV1S0 SFT0 allL0301000 LR acc wvhalf  KMeans CA50 MC1 MS0.4 ATS1000 DS100",
                       "wiki300LSTMCstate5010kCV1S0 SFT0 allL0301000 LR acc wvhalf  KMeans CA50 MC1 MS0.4 ATS500 DS100"]
    max_features_a = [10000] # Was 20,000 S
    maxlen_a = [200]  # cut texts after this number of words (among top max_features most common words) # L
    batch_size_a = [32] # M
    epochs_a = [16] #15,30,10 # L
    dropout_a = [0.0] # L
    recurrent_dropout_a = [0.0] # S
    embedding_size_a = [16] # S
    lstm_size_a = [50] # S
    learn_rate_a = [0.01] # S
    scale_amount_a = [1,2,5,10,20,40,80,160,320,640]
    scale_amount_2_a = [1000]
    stateful =False
    forget_bias = True
    dev = True
    use_all = False
    iLSTM = True
    iLSTM_t_model = False
    rewrite = False
    use_lr = False
    test = False
    tensorBoard = True
    use_wv = True
    trainable = True
elif dataset == 1:
    n_clusters_fn_a = ["zeros"]
    max_features_a = [10000] # Was 20,000 S
    maxlen_a = [50]  # cut texts after this number of words (among top max_features most common words) # L
    batch_size_a = [32] # M
    epochs_a = [8] #15,30,10 # L
    dropout_a = [0.0] # L
    recurrent_dropout_a = [0.0] # S
    embedding_size_a = [16] # S
    lstm_size_a = [10] # S
    learn_rate_a = [0.01] # S
    scale_amount_a = [1000]
    scale_amount_2_a = [1000]
    stateful =False
    forget_bias = False
    dev = True
    use_all = False
    iLSTM = False
    iLSTM_t_model = False
    rewrite = True
    use_lr = False
    test = False
    tensorBoard = True
    use_wv = True
    trainable = True
elif dataset == 2:
    n_clusters_fn_a = ["lstm128CCV1S0 SFT0 allL03018836 LR kappa KMeans CA128 MC1 MS0.4 ATS128 DS128",
                       "lstm128CCV1S0 SFT0 allL03018836 LR kappa KMeans CA128 MC1 MS0.4 ATS1000 DS256"]
    max_features_a = [10000] # Was 20,000 S
    maxlen_a = [1000]  # cut texts after this number of words (among top max_features most common words) # L
    batch_size_a = [32] # M
    epochs_a = [16] #15,30,10 # L
    dropout_a = [0.0] # L
    recurrent_dropout_a = [0.0] # S
    embedding_size_a = [16] # S
    lstm_size_a = [128] # S
    learn_rate_a = [0.01] # S
    scale_amount_a = [1]
    scale_amount_2_a = [1000]
    stateful =False
    forget_bias = True
    dev = True
    use_all = False
    iLSTM = True
    iLSTM_t_model = False
    rewrite = False
    use_lr = False
    test = False
    tensorBoard = True
    use_wv = True
    trainable = True

use_L2_a = [0.1]

if use_wv is False:
    trainable = True

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
all_params.append(scale_amount_a)
all_params.append(scale_amount_2_a)
all_params.append(n_clusters_fn_a)
all_params.append(use_L2_a)


max_index_a = np.zeros(len(all_params), dtype="int")

# 0 = IMDB, 1 = SST

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

word_vector_fn = "google_news_no_bigram_IMDB_words"

import_model = None#"wvMF10000 ML300 BS25 FBTrue DO0.2 RDO0.1 E128 ES300LS50 UAFalse SFFalse iLTrue rTFalse lrFalse sA1 sA2100 wiki"


if import_model is not None:
    variables = import_model.split()
    max_features = int(variables[1])
    maxlen = int(variables[2][2:])
    batch_size = int(variables[3][2:])
    epochs = int(variables[7][1:])
    forget_bias = bool(variables[4][2:])
    dropout = float(variables[5][2:])
    recurrent_dropout = float(variables[6][3:])
    embedding_size = int(variables[8].split("L")[0][2:])
    try:
        lstm_size = int(variables[8].split("L")[1][1:])
    except ValueError:
        lstm_size =  int(variables[8].split("L")[1][1:].split(".")[0])
        print(".Txt detected, split on '.', result is: " + str(lstm_size))
    try:
        stateful = ast.literal_eval(variables[10][2:])
    except IndexError:
        print("Stateful not included, set to default false value")
        stateful = False
    try:
        iLSTM = ast.literal_eval(variables[11][2:])
    except IndexError:
        print("iLSTM not included, set to default false value")
        iLSTM = False
    try:
        iLSTM_t_model = ast.literal_eval(variables[12][2:])
    except IndexError:
        print("iLSTM not included, set to default false value")
        iLSTM_t_model = False
    try:
        use_lr = bool(variables[13][2:])
    except IndexError:
        print("use_lr not included, set to default false value")
        use_lr = False
    try:
        scale_amount = float(variables[14][2:])
    except IndexError:
        print("scale_amt not included, set to default false value")
        scale_amount = 1
    try:
        scale_amount = float(variables[15][3:])
    except ValueError:
        print("scale_amt not included, set to default false value")
        scale_amount = 1
    except IndexError:
        print("scale_amt not included, set to default false value")
        scale_amount = 1


print('Loading data...')

# We need to load the IMDB dataset. We are constraining the dataset to the top 5,000 words. We also split the dataset into train (50%) and test (50%) sets.
if dataset == 0:
    data_path = "../data/sentiment/lstm/"
elif dataset == 1:
    data_path = "../data/sst/lstm/"
elif dataset == 2:
    data_path = "../data/newsgroups/lstm/"

all_lens = 0
for z in range(len(all_params)):
    all_lens += len(all_params[z])
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
        learn_rate = all_params[8][max_index_a[8]]
        scale_amount = all_params[9][max_index_a[9]]
        scale_amount_2 = all_params[10][max_index_a[10]]
        n_clusters_fn = all_params[11][max_index_a[11]]
        use_L2 = all_params[12][max_index_a[12]]

    print(max_features, maxlen, batch_size, epochs, dropout, recurrent_dropout, embedding_size, lstm_size, stateful, iLSTM,
          n_clusters_fn, use_L2)
    max_index = 0
    max_acc = 0
    for j in range(len(all_params[i])):
        if import_model is None:
            if i == 0:
                max_features = all_params[0][j]
            if i == 1:
                maxlen = all_params[1][j]
            if i == 2:
                batch_size = all_params[2][j]
            if i == 3:
                epochs = all_params[3][j]
            if i == 4:
                dropout = all_params[4][j]
            if i == 5:
                recurrent_dropout = all_params[5][j]
            if i == 6:
                embedding_size = all_params[6][j]
            if i == 7:
                lstm_size = all_params[7][j]
            if i == 8:
                learn_rate = all_params[8][j]
            if i == 9:
                scale_amount = all_params[9][j]
            if i == 10:
                scale_amount_2 = all_params[10][j]
            if i == 11:
                n_clusters_fn = all_params[11][j]
            if i == 12:
                use_L2 = all_params[12][j]
        else:
            j = len(all_params[i])-1
        i_output_name = n_clusters_fn + " tdev"
        x_train, y_train, x_test, y_test, word_index, y_cell_test, y_cell_train = \
            manage.getData(max_features, iLSTM, i_output_name, dev, use_all, test, maxlen, dataset, data_path)

        embedding_matrix, wv_size, wi_size = manage.getEmbeddingMatrix(word_vector_fn, word_index, dataset, data_path[:-5])

        embedding_size = wv_size

        if np.amax(embedding_matrix) <= 0:
            raise Exception("Word embeddings not initialized")

        if import_model is None:
            file_name = ""
            if test:
                file_name += "TEST"
                epochs = 1
            if use_wv:
                file_name += "wv"
            if trainable:
                file_name += "Train"
            file_name = file_name + "MFTraFAdr1337mse"+str(dataset)+ " " + str(max_features) + " ML" + str(maxlen) + " BS" + str(
                batch_size) + " FB" + str(
                forget_bias) + " DO" \
                        + str(dropout) + " RDO" + str(recurrent_dropout) + " E" + str(epochs) + " ES" + str(
                embedding_size) + "LS" + \
                        str(lstm_size) + " UA" + str(use_all) + " SF" + str(stateful) + " iL" + str(iLSTM) + " rT" + \
                        str(iLSTM_t_model) + " lr" + str(use_lr) + " sA" + str(scale_amount) + " " + n_clusters_fn[:4]
            if iLSTM:
                fn_to_add = ""
                for i in range(len(n_clusters_fn.split())):
                    if n_clusters_fn.split()[i][:2] == "CA" or n_clusters_fn.split()[i][:3] == "ATS" \
                            or "kappa" in n_clusters_fn.split()[i]\
                            or "acc" in n_clusters_fn.split()[i]:
                        fn_to_add = fn_to_add + " " +  n_clusters_fn.split()[i]
                file_name = file_name + fn_to_add
        else:
            file_name = import_model

        print("FN HERE:", file_name)

        vector_fn = data_path + "vectors/" + file_name + " L" + str(1)
        model_fn = data_path + "model/" + file_name
        state_fn = data_path + "states/" + file_name + " HState"
        final_state_fn = data_path + "states/" + file_name + " FState"

        model = manage.getModel(import_model, max_features, maxlen, embedding_size, lstm_size, forget_bias, recurrent_dropout,
             dropout, stateful, iLSTM, scale_amount, x_train, y_train, x_test, y_test, y_cell_train, y_cell_test, wv_size,
             wi_size, batch_size, epochs, embedding_matrix, model_fn, file_name, dataset, use_wv, data_path, trainable)

        manage.saveModel(model, model_fn)

        if iLSTM:
            y_names = import1dArray(data_path + "n_clusters/"+n_clusters_fn+".txt", "s")
        else:
            y_names = None
        if dev:
            file_name = file_name + " dev"
        else:
            file_name = file_name + " test"

        score_fn = data_path + "score/" + file_name
        acc, score, f1 = manage.saveScores(model, score_fn, dev, x_test, y_test, rewrite, batch_size, y_cell=y_cell_test,
                                           y_names=y_names, dataset=dataset)
        if not iLSTM:
            manage.saveVectors(model, vector_fn, x_train)
        manage.saveState(model, state_fn, x_train, lstm_size, maxlen, final_state_fn, iLSTM)
        """
        if iLSTM and not iLSTM_t_model:
            file_name = "wvMF" + str(max_features) + " ML" + str(maxlen) + " BS" + str(batch_size) + " FB" + str(
                forget_bias) + " DO" \
                        + str(dropout) + " RDO" + str(recurrent_dropout) + " E" + str(epochs) + " ES" + str(
                embedding_size) + "LS" + \
                        str(lstm_size) + " UA" + str(use_all) + " SF" + str(stateful) + " iL" + str(iLSTM) + " rTTrue" + \
                        " lr" + str(use_lr)+ " sA" + str(scale_amount)+ " sA2" + str(scale_amount_2)+ " " + n_clusters_fn[:4]

            vector_fn = data_path + "vectors/" + file_name + " L" + str(1)
            model_fn = data_path + "model/" + file_name
            acc_fn = data_path + "score/" + file_name + " acc"
            score_fn = data_path + "score/" + file_name + " score"
            state_fn = data_path + "states/" + file_name + " HState"
            final_state_fn = data_path + "states/" + file_name + " FState"

            if os.path.exists(model_fn) is False:
                model.compile(loss=['binary_crossentropy', 'mse'],
                              optimizer='adam',
                              metrics=['accuracy'], loss_weights=[1.0, 1.0 / scale_amount_2])
                print('Train...')

                model.fit(x_train, [y_train, y_cell_train],
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, [y_test, y_cell_test]))

                manage.saveModel(model, model_fn)
            else:
                model = keras.models.load_model(model_fn)
            acc, score, f1 = manage.saveScores(model, score_fn, dev, x_test, y_test, rewrite, batch_size, y_cell=y_cell_test, y_names=y_names)
        """
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
        if i == 0:
            max_features = all_params[0][max_index]
        if i == 1:
            maxlen = all_params[1][max_index]
        if i == 2:
            batch_size = all_params[2][max_index]
        if i == 3:
            epochs = all_params[3][max_index]
        if i == 4:
            dropout = all_params[4][max_index]
        if i == 5:
            recurrent_dropout = all_params[5][max_index]
        if i == 6:
            embedding_size = all_params[6][max_index]
        if i == 7:
            lstm_size = all_params[7][max_index]
        np.savetxt(data_path + "score/" + "max vals" + str(i) + " acc", max_index_a)
        print(all_params)
        np.save(data_path + "score/" + "all params" + str(i) +  " acc", all_params)


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




