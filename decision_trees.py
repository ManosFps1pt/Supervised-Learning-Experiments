import numpy as np


def entropy(y):
    count = np.unique(y, return_counts=True)
    p = count[1] / len(y)
    log = np.log2(p)
    res = -np.sum(p * log)
    return res

print(entropy([0, 1, 2]))
print(np.log2(3))
