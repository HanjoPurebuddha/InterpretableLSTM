from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
from keras import backend as K
import numpy as np

# 10 clusters
# 5 entities

binary_cluster_array = np.random.randint(low=0, high=2, size=(10,5), dtype=np.int64)
yT = np.random.random(size=(10,5))
yP = np.random.random(size=(10,5))

#So in the case of 64 clusters, we have 64 outputs, with one additional output for the supervised objective, making 65 outputs. Each one of the cluster outputs has the loss function of:  log P(v_d |C), or log (1- P(v_d |C)), dependent on whether or not the document (in this case a sequence) contains that particular word, which we would check, I assume, using a binary input that represents if each document has a word present in a cluster, and this bag-of-words/clusters is given to the loss function for this purpose.

#note that you should indeed use negative log values, e.g. if d contains one of the cluster terms then we want to maximise P(v_d|C) which is achieved by maximising log P(v_d|C) and thus by minimising -log P(v_d|C).

#yTrue = probability, yPred = hidden state
def logLoss(yTrue,yPred):
    logs = np.zeros(len(yTrue), dtype=np.float64)
    for y in range(len(yTrue)):
        if binary_cluster_array[0][y] == 1:
            print(binary_cluster_array[0][y], yPred[y], -np.log(yPred[y]))
            logs[y] = -np.log(yPred[y])
        else:
            print(binary_cluster_array[0][y], yPred[y], -np.log(yPred[y]))
            logs[y] = -np.log(1-yPred[y])
    sum = np.sum(logs)
    print("np sum", sum)
    return K.sum(logs)

def logLossNew(yTrue, yPred):
    return K.sum(-K.log(yTrue - yPred))
#print(logLossNew(yT[0], yP[0]))

def logLossSupervised(yTrue, yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))


from keras.models import *
from keras.layers import *

features = 10
dimensions = 3
timesteps = 5

y = np.random.randint(low=0, high=2, size=(features), dtype=np.int64)
y_cell = np.random.randint(low=0, high=2, size=(features, dimensions), dtype=np.int64)
x = np.random.random(size=(features, timesteps))

input_layer = Input(shape=(timesteps,))#
input_sequence = Embedding(input_dim=1, output_dim =dimensions)(input_layer)
h_l1, h_l2, cell_state = LSTM(dimensions, return_state=True)(input_sequence)
output_layer = Dense(1)(h_l1)

model = Model(input_layer, [output_layer, cell_state])

model.compile(loss=['mse', 'binary_crossentropy'],
                          optimizer='adam',
                          metrics=['accuracy'])

class_weight_y = {}
class_weight_y_cell = {}
n_classes = len(y_cell[0])
scale = 1

class_weight_y[0] = n_classes * scale
class_weight_y[1] = n_classes * scale
for i in range(n_classes):
    class_weight_y_cell[i] = 1

model.fit(x, [y, y_cell], class_weight=[class_weight_y, class_weight_y_cell])
