import numpy as np
import os
import chardet
import locale
import read_text as rt
file_structure = "../data/sentiment/wordvectors/data/"

# test

test_pos = file_structure + "orig/test/pos/"
test_neg = file_structure + "orig/test/neg/"
train_pos = file_structure + "orig/train/pos/"
train_neg = file_structure + "orig/train/neg/"
train_unsup = file_structure + "orig/train/unsup/"
folder_names = [test_pos, test_neg, train_pos, train_neg, train_unsup]
compiled_names = ["test_pos", "test_neg", "train_pos", "train_neg", "unsup"]

for i in range(len(folder_names)):
    fns = rt.getFns(folder_names[i])
    arrays = []
    for fn in fns:
        arrays.append(rt.importArray(folder_names[i] + fn))
    rt.writeArrays(arrays, file_structure + "compiled/" + compiled_names[i] + ".txt")

