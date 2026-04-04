import numpy as np


def entropy(y):
    count = np.unique(y, return_counts=True)
    p = count[1] / len(y)
    log = np.log2(p)
    res = -np.sum(p * log)
    return res


def information_gain(x_colum: np.array, y: np.array):
    ent_y = entropy(y)
    for x in np.unique(x_colum):
        mask = x_colum == x
        label_y = y[mask]
        ent_y -= (len(label_y) / len(x_colum)) * entropy(label_y)
    return ent_y


def tree(data: np.array, labels: np.array):
    max_gain = 0
    arr_idx = None
    for idx, i in enumerate(data):
        inf_gain = information_gain(i, labels)
        if inf_gain > max_gain:
            max_gain = inf_gain
            arr_idx = idx
    return arr_idx


# Case 1: feature perfectly separates the labels
print(information_gain(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])))

# Case 2: feature tells you nothing
print(information_gain(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])))

data = np.array([
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
])
labels = np.array([1, 0, 1, 0])
print(tree(data, labels))
