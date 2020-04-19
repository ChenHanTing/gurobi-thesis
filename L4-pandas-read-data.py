import pandas as pd
import numpy as np

df = pd.read_excel('ex02.xlsx')
print(df)

# 可參考：https://realpython.com/pandas-groupby/
dfg = df.groupby('種類')

list = []

for item, group in dfg:
    list.append(np.delete(group.to_numpy(), slice(2), 1))

for group in list:
    for i in range(group.shape[0]):
        for j in range(group.shape[1]):
            print([i, j])

# npa = df.to_numpy()
# print(npa)

# # 可參考： https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
# npar = npa.reshape(npa.shape[0], 2, 2)
# print(npar)

# a = np.arange(12).reshape(3, 4)
# # [[ 0  1  2  3]
# #  [ 4  5  6  7]
# #  [ 8  9 10 11]]

# # print(np.delete(a, [0, 3], 1))
# # print(np.delete(npa, slice(2), 1))
