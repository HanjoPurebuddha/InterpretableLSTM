import numpy as np
import os

def makeAndLoad(functToMake, fn_to_save):
    if os.path.exists(fn_to_save):
        val = np.load(fn_to_save)
        return val
    val = functToMake()
    return val

