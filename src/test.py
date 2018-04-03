
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



import tensorflow as tf


from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from numpy import array
import keras.backend as K


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return 1 / x


# Extends Keras' TensorBoard callback to include the Precision-Recall summary pl
b_fn = "../data/sentiment/lstm/b_clusters/wiki300LSTMCstate5010kCV1S0 SFT0 allL0301000LRacc wvhalf  KMeans CA50 MC1 MS0.4 ATS500 DS100 tdev.npy"

n_fn = "../data/sentiment/lstm/n_clusters/wiki300LSTMCstate5010kCV1S0 SFT0 allL0301000LRacc wvhalf  KMeans CA50 MC1 MS0.4 ATS500 DS100.txt"
import read_text as rt
y_cell = np.load(b_fn)


ones_cell = np.load("../data/sentiment/lstm/states/"
                    "wvMFTraFAdr1337 10000 ML200 BS32 FBTrue DO0.2 RDO0.1 E16 ES300LS50 UAFalse SFFalse iLTrue rTFalse lrFalse sA1000 ones FState.npy")


random_y_cell = np.zeros(shape=(128,18846), dtype=np.int64)
ones = 0
"""
for yc in range(len(y_cell)):
    for i in range(len(y_cell[yc])):
        y_cell[yc][i] = 100
    
    for i in range(len(np.nonzero(y_cell[yc])[0])):
        id = 0
        while random_y_cell[yc][id] == 1:
            id = np.random.randint(low=0, high=len(y_cell[yc]))
        random_y_cell[yc][id] = 1
    """

np.save("../data/newsgroups/lstm/b_clusters/zeros tdev.npy", random_y_cell)



y_cell_pred = np.load(cell_state_fn).transpose()


y_names = np.asarray(rt.importArray(n_fn, encoding="cp1252"))

y_cell = y_cell.transpose()
y_cell = y_cell[:len(y_cell_pred)]
y_cell = y_cell.transpose()

y_cell_pred = y_cell_pred.transpose()

y_cell_pred_classes = np.empty(len(y_cell_pred), dtype=np.object)
for c in range(len(y_cell_pred)):
    y_cell_pred_classes[c] = np.where(y_cell_pred[c] > 0.5, 1, y_cell_pred[c])
    y_cell_pred_classes[c] = np.where(y_cell_pred_classes[c] <= 0.5, 0, y_cell_pred_classes[c])
    y_cell_pred_classes[c] = y_cell_pred_classes[c].astype(np.int64)

y_cell_pred_classes = y_cell_pred_classes.transpose()








highest_acc = np.zeros(len(y_cell))

if y_cell is not None:
    max_f1s = []
    for i in range(len(y_cell_pred_classes)):
        max_f1 = 0
        for c in range(len(y_cell_pred_classes)):
            if c != 0:
                current = y_cell_pred_classes[c]
                y_cell_pred_classes[c] = y_cell_pred_classes[c-1]
                y_cell_pred_classes[c-1] = current
            cell_acc = accuracy_score(y_cell[c], y_cell_pred_classes[c])
            cell_f1 = f1_score(y_cell[c], y_cell_pred_classes[c], average="binary")
            #print(y_names[c], "acc", cell_acc, "f1", cell_f1)
            if max_f1 < cell_f1:
                max_f1 = cell_f1
        print(y_names[i], max_f1)
        max_f1s.append(max_f1)
    print(max_f1s)

dict = {}
dict["aa"] = 0
dict["bb"] = 1
dict["cc"] = 3

origin = "../data/sst/raw/"
import read_text as rt

hstate_ilstm = np.load("../data/sentiment/lstm/states/"
                       "wvMFTraFAd 10000 ML200 BS32 FBTrue DO0.2 RDO0.1 E16 ES300LS50 UAFalse SFFalse iLTrue rTFalse lrFalse sA1 wiki FState.npy")
hstate_lstm = np.load("../data/sentiment/lstm/states/"
                       "wvMFTraFAd 10000 ML200 BS32 FBTrue DO0.2 RDO0.1 E16 ES300LS50 UAFalse SFFalse iLFalse rTFalse lrFalse sA1 sA2100 wiki FState.npy")
corpus[0][1] = words[0]
corpus[0] = np.insert(corpus[0], 1+1, words[1:])
print(corpus)

exit()
# Approach 1:
# set up fields, common NLP types that can convert into tensors (e.g. a sentence composed of indexes of words in a vocab)
# tokenize using spacy, not the default str.split()
TEXT = data.Field(use_vocab=True, sequential=True, tokenize="spacy")
# We aren't dealing with sequential data for the labels, e.g. a sentence, so sequential=False
LABEL = data.Field(use_vocab=True, sequential=False)

# make splits for data
# train, test, val are the data split into those categories
# each one is an array of "examples", which are label/text pairs where the text is tokenized
# train subtrees is when you include in the training data all subtrees of the original sentence reviews. shown to increase \v/
# performance on the fine-grained, need to check if it increases it on the binary
# fine_grained is the fine grained task, with 5 different sentiments, we want binary for this experiment
# there are 5 labels by default, very negative, negative, neutral, positive, very positive.
# We want the binary, w/o subtrees, and neutrals removed. For unsupervised, we would want neutrals included
train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=False, train_subtrees=False,
    filter_pred=lambda ex: ex.label != 'neutral')

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary, shared between the text/label fields
# Only allow words within the word vector vocab of this word vector
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

# make iterator for splits, convert words from vocab into indexes
# set device to -1 to use cpu
train_iter = data.BucketIterator(train, batch_size=len(train.examples), device=-1)

x_train = None
y_train = None


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)

train_dl = BatchWrapper(train_iter, "text", "labels")

for d in iter(train_dl):
    print(d)

print("?")
#model.word_em.weight.data = train_dataset.fields["comment_text"].vocab.vectors


# Don’t worry. This step is still very easy to handle. We can obtain the Vocab object easily from the Field
# (there is a reason why each Field has its own Vocab class, because of some pecularities of Seq2Seq model like
#  Machine Translation, but I won’t get into it right now.).
# We use PyTorch’s nice Embedding Layer to solve our embedding lookup problem:
"""
vocab = TEXT.vocab
self.embed = nn.Embedding(len(vocab), emb_dim)
self.embed.weight.data.copy_(vocab.vectors)
"""
"""
# Approach 2:
TEXT.build_vocab(train, vectors=[GloVe(name='840B', dim='300'), CharNGram(), FastText()])
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter, val_iter, test_iter = datasets.SST.iters(batch_size=4)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)

# Approach 3:
f = FastText()
TEXT.build_vocab(train, vectors=f)
TEXT.vocab.extend(f)
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))
print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

train_iter, val_iter, test_iter = datasets.SST.iters(batch_size=4)

# print batch information
batch = next(iter(train_iter))
print(batch.text)
print(batch.label)
"""