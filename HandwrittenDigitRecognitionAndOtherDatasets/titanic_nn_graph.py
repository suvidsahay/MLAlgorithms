import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import neural_networks_cost as neural_network

def normalise(train_df, test_df):
    numerical = ['remainder__Age', 'remainder__Siblings/Spouses Aboard', 'remainder__Parents/Children Aboard', 'remainder__Fare']
    max_values = train_df[numerical].max()
    min_values = train_df[numerical].min()

    ranges = max_values - min_values
    ranges[ranges == 0] = 1

    train_df[numerical] = 2 * (train_df[numerical] - min_values) / ranges - 1
    test_df[numerical] = 2 * (test_df[numerical] - min_values) / ranges - 1

    return train_df, test_df

def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def split(df, output):
    df_i = [None] * 10

    for j in range(10):
        df_i[j] = df[df[output] == j]

    # print(len(df_0), len(df_1))

    # print(df[output].value_counts()) 

    train_df = pd.DataFrame()   

    for j in range(10):
        train_df = pd.concat([train_df, df_i[j].sample(frac = 0.8)])
    
    # print(len(test_df))
    test_df = df.drop(train_df.index)

    return train_df, test_df

def divide (a, b):
    if b == 0:
        return np.nan
    else:
        return a / b

df = pd.read_csv('datasets/titanic.csv')
df = df.drop(columns=['Name'])


attr = ['Pclass', 'Sex']
shuffle(df)

transformer = make_column_transformer(
    (OneHotEncoder(), attr),
    remainder='passthrough')

transformed = transformer.fit_transform(df)

df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
output_class = 'remainder__Survived'
r_lambda = 0.2
alpha = 1


train_df, test_df = split(df, output_class)
print(train_df.columns)

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
nn = neural_network.NeuralNetwork(train_df, 1, [2], r_lambda, alpha, y_train)
J = nn.getCost(test_df, y_test)


 


