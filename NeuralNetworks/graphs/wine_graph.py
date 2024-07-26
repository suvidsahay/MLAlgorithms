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

def normalise(train_df, test_df):
    max_values = train_df.max()
    min_values = train_df.min()

    train_df = 2 * (train_df - min_values) / (max_values - min_values) - 1
    test_df = 2 * (test_df - min_values) / (max_values - min_values) - 1

    return train_df, test_df

def split(df, output):
    df_1 = df[df[output] == 1]
    df_2 = df[df[output] == 2]
    df_3 = df[df[output] == 3]


    train_df = pd.concat([df_1.sample(frac = 0.8), df_2.sample(frac=0.8), df_3.sample(frac=0.8)])
    test_df = df.drop(train_df.index)

    return train_df, test_df

def divide (a, b):
    if b == 0:
        return np.nan
    else:
        return a / b

df = pd.read_csv("datasets/hw3_wine.csv", delim_whitespace=True)

attr = [
    ('1', True), 
    ('2', True), 
    ('3', True), 
    ('4', True), 
    ('5', True), 
    ('6', True), 
    ('7', True), 
    ('8', True), 
    ('9', True), 
    ('10', True), 
    ('11', True), 
    ('12', True), 
    ('13', True)
]

    

df = shuffle(df)

r_lambda = 0.01
alpha = 1
output_class='class'

train_df, test_df = split(df, output_class)
print(train_df[output_class].value_counts())
print(test_df[output_class].value_counts())
y_train = train_df[output_class]
# print(np.unique(y_train, return_counts=True))
y_train = np.zeros((len(train_df), y_train.nunique()))
for i in range(len(train_df)):
    # print(train_df[output_class].iloc[i])
    y_train[i][int(train_df[output_class].iloc[i]) - 1] = 1

train_df = train_df.drop(columns=[output_class])

y_test = test_df[output_class]
y_test = np.zeros((len(test_df), y_test.nunique()))
for i in range(len(test_df)):
    # print(train_df[output_class].iloc[i])
    y_test[i][int(test_df[output_class].iloc[i]) - 1] = 1

test_df = test_df.drop(columns=[output_class])

train_df, test_df = normalise(train_df, test_df)

# print(train_df.head())
nn = neural_network.NeuralNetwork(train_df, 1, [8], r_lambda, alpha, y_train)
J = nn.getCost(test_df, y_test)


fig, ax = plt.subplots()

ax.plot(list(range(len(J))), J, label='Cost')

ax.set_title('Cost function vs number of training samples')
ax.set_xlabel('Number of training sample')
ax.set_ylabel('Cost')

ax.legend()

plt.grid(True) 
plt.show()


