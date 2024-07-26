import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

import neural_network_cost as neural_network


def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def split(df, output):
    df_0 = df[df[output] == 0]
    df_1 = df[df[output] == 1]


    train_df = pd.concat([df_0.sample(frac = 0.8), df_1.sample(frac=0.8)])
    test_df = df.drop(train_df.index)

    return train_df, test_df

def normalise(train_df, test_df):
    max_values = train_df.max()
    min_values = train_df.min()

    train_df = 2 * (train_df - min_values) / (max_values - min_values) - 1
    test_df = 2 * (test_df - min_values) / (max_values - min_values) - 1

    return train_df, test_df

def divide (a, b):
    if b == 0:
        return np.nan
    else:
        return a / b


df = pd.read_csv("datasets/hw3_cancer.csv", delim_whitespace=True)

attr = [
    ('Clump_Thickness',True),
    ('Cell_Size_Uniformity', True),
    ('Cell_Shape_Uniformity', True),
    ('Marginal_Adhesion', True),
    ('Single_Epi_Cell_Size', True),
    ('Bare_Nuclei', True),
    ('Bland_Chromatin', True),
    ('Normal_Nucleoli', True),
    ('Mitoses', True)
]

    

df = shuffle(df)


r_lambda = 5
alpha = 0.5
output_class='Class'

train_df, test_df = split(df, output_class)
print(train_df[output_class].value_counts())
print(test_df[output_class].value_counts())
y_train = train_df[output_class]
# print(np.unique(y_train, return_counts=True))
y_train = np.zeros((len(train_df), y_train.nunique()))
for i in range(len(train_df)):
    # print(train_df[output_class].iloc[i])
    y_train[i][int(train_df[output_class].iloc[i])] = 1

train_df = train_df.drop(columns=[output_class])

y_test = test_df[output_class]
y_test = np.zeros((len(test_df), y_test.nunique()))
for i in range(len(test_df)):
    # print(train_df[output_class].iloc[i])
    y_test[i][int(test_df[output_class].iloc[i])] = 1

test_df = test_df.drop(columns=[output_class])

train_df, test_df = normalise(train_df, test_df)

# print(train_df.head())
nn = neural_network.NeuralNetwork(train_df, 1, [16], r_lambda, alpha, y_train)
J = nn.getCost(test_df, y_test)


fig, ax = plt.subplots()

ax.plot(list(range(len(J))), J, label='Cost')

ax.set_title('Cost function vs number of training samples')
ax.set_xlabel('Number of training sample')
ax.set_ylabel('Cost')

ax.legend()

plt.grid(True) 
plt.show()


