import parse_binary_treebank as pbt
import read_text as rt
import numpy as np
import scipy.sparse as sp

origin = "../data/sst/raw/"

vocab = rt.load_dict(origin+"vocab.dict")
reversed_vocab = rt.load_dict(origin+"reversed_vocab.dict")


x_train, x_test, x_dev, y_train, y_test, y_dev = pbt.loadProcessedSplits()

lowest_amt = 0
highest_amt = 5
classification = "all"

save_origin = "/mnt/62423FE6423FBD9B/PhD/Code/Paper 2/data/"
all_fn = save_origin + "sst/bow/frequency/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification+".npz"
all_fn_binary = save_origin + "sst/bow/binary/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification +".npz"
word_fn = save_origin + "sst/bow/names/" + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification + ".txt"

vectors = np.concatenate((x_train, x_dev, x_test), axis=0)
classes = np.concatenate((y_train, y_dev, y_test), axis=0)

word_list = list(vocab.keys())

tf = np.zeros(shape=(len(vectors), len(word_list)), dtype=np.int32)
tf_binary = np.zeros(shape=(len(vectors), len(word_list)), dtype=np.int32)

for ds in range(len(vectors)): # d for document sequence
    for wi in range(len(vectors[ds])): # every word id in the sequence
        tf[ds][vectors[ds][wi]] += 1
        tf_binary[ds][vectors[ds][wi]] = 1

tf = tf.transpose()
tf_binary = tf_binary.transpose()
ind_to_delete = []
for wi in range(len(tf)):
    if len(np.nonzero(tf[wi])[0]) <= highest_amt:
        print("0 total", reversed_vocab[wi])
        ind_to_delete.append(wi)
print("to delete", len(ind_to_delete))
tf = np.delete(tf, ind_to_delete, axis=0)
tf_binary = np.delete(tf_binary, ind_to_delete, axis=0)
word_list = np.delete(word_list, ind_to_delete, axis=0)

classification = "binary"
rt.writeArray(word_list, word_fn)

tf_binary = sp.csr_matrix(tf_binary)
tf = sp.csr_matrix(tf)

sp.save_npz(all_fn_binary, tf_binary)
sp.save_npz(all_fn, tf)
#np.save(all_fn_binary, tf_binary)
#np.save(all_fn, tf)