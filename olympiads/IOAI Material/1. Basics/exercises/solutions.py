#%%
import numpy as np
import pandas as pd
import random
np.random.seed(42)
#%%
# Solution of problem 1:
X = np.random.randn(12,5)
X_centered = X - X.mean(axis=0)
print(X_centered.mean(axis=0))
X_row_normalized = X / np.linalg.norm(X, keepdims=True, axis=1)
print(np.linalg.norm(X_row_normalized, axis=1))
X_relu = np.where(X>0, X, 0)
print(X_relu)
positive_first_feature_rows = X[X[:,0]>0]
print(positive_first_feature_rows)
#%%
X1 = np.random.randn(7, 4, 6)
X2 = np.random.randn(7, 6, 3)
print((X1 @ X2).shape)
print(np.einsum("xij,xjy->xiy", X1, X2).shape)
# %%
M = np.random.randn(8, 4)
# %%
# problem 2
n = 100_000
table = pd.DataFrame({
    "division": np.random.choice("AB", n),
    "NPOPR": np.random.randint(20, 250, n),
    "RP": np.random.rand(n) * 6
})