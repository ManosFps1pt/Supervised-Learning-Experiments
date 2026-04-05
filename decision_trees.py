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

def split_with_greatest_inf_gain(data, labels):
    max_gain = 0
    arr_idx = None
    for idx, i in enumerate(data):
        inf_gain = information_gain(i, labels)
        if inf_gain > max_gain:
            max_gain = inf_gain
            arr_idx = idx
    return arr_idx

def tree(data: np.array, labels: np.array, depth=0, max_depth=3):
    if entropy(labels) == 0:
        return labels[0]
    if depth == max_depth:
        return np.bincount(labels).argmax()
    idx = split_with_greatest_inf_gain(data, labels)
    best_feature = data[idx]
    subtrees = {}
    for x in np.unique(best_feature):
        mask = best_feature == x
        sub_data = data[:, mask]
        sub_labels = labels[mask]
        subtrees[x] = tree(sub_data, sub_labels, depth + 1, max_depth)
    return {'feature': idx, 'branches': subtrees}


# Case 1: feature perfectly separates the labels
print(information_gain(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])))

# Case 2: feature tells you nothing
print(information_gain(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])))

data = np.array([
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
])
labels = np.array([1, 0, 1, 0])
print(f'tree: {tree(data, labels)}')
