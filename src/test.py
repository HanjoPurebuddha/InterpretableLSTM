from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras import Model
import numpy as np

input_tensor = Input(shape=(None,), dtype='int32')
embedding = Embedding(10, 100, mask_zero=True)(input_tensor)
hidden = LSTM(2)(embedding)
out = Dense(1, activation='sigmoid')(hidden)
model = Model(input_tensor, out)

target_layer = model.layers[-2]
target_layer.return_sequences = True
outputs = target_layer(target_layer.input)
m = Model(model.input, outputs)

X_test = np.array([[1, 3, 2, 0, 0]])

print(m.predict(X_test))