import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

import neural_network


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



def k_fold_partition(df, k, i, output):
    df_1 = df[df[output] == 1]
    df_2 = df[df[output] == 2]
    df_3 = df[df[output] == 3]

    # print(df['class'].value_counts())    

    test_df = pd.concat([df_3[int((len(df_3) * i) / k) : int((len(df_3) * (i + 1)) / k)], 
                         df_1[int((len(df_1) * i) / k) : int((len(df_1) * (i + 1)) / k)],
                         df_2[int((len(df_2) * i) / k) : int((len(df_2) * (i + 1)) / k)]])
    train_df = df.drop(test_df.index)

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
k_fold = 10
r_lambda = 0.01
alpha = 0.5
output_class = 'class'
for layers in range(0, 5, 1):
    for neurons in (2 ** x for x in range(1, 5, 1)):
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        total_confusion_matrix = pd.DataFrame(np.zeros((3, 3)), index=[1, 2, 3], columns=['pred_1', 'pred_2', 'pred_3'])

        for k in range(k_fold):

            train_df, test_df = k_fold_partition(df, k_fold, k, output_class)
            y_train = train_df[output_class]
            # print(np.unique(y_train, return_counts=True))
            y_train = np.zeros((len(train_df), y_train.nunique()))

            for i in range(len(train_df)):
                # print(train_df[output_class].iloc[i])
                y_train[i][int(train_df[output_class].iloc[i]) - 1] = 1
            
            train_df = train_df.drop(columns=[output_class])

            # print(train_df.head())
            y_test = test_df[output_class]
            test_df = test_df.drop(columns=output_class)

            # print(train_df.shape)
            # print(test_df.shape)

            train_df, test_df = normalise(train_df, test_df)


            nn = neural_network.NeuralNetwork(train_df, layers, [neurons] * layers, r_lambda, alpha, y_train)
            nn.fit()

            confusion_matrix = pd.DataFrame(np.zeros((3, 3)), index=[1, 2, 3], columns=['pred_1', 'pred_2', 'pred_3'])

            for i in range(len(test_df)):
                output = nn.test(test_df.iloc[i])
                expected = y_test.iloc[i] - 1
                if output == expected:
                    if output == 0:
                        confusion_matrix.loc[1, 'pred_1'] += 1
                    elif output == 1:
                        confusion_matrix.loc[2, 'pred_2'] += 1
                    else:
                        confusion_matrix.loc[3, 'pred_3'] += 1
                else:
                    if output == 0:
                        if expected == 1:
                            confusion_matrix.loc[2, 'pred_1'] += 1
                        else:
                            confusion_matrix.loc[3, 'pred_1'] += 1
                    elif output == 1:
                        if expected == 0:
                            confusion_matrix.loc[1, 'pred_2'] += 1
                        else:
                            confusion_matrix.loc[3, 'pred_2'] += 1
                    elif output == 2:
                        if expected == 0:
                            confusion_matrix.loc[1, 'pred_3'] += 1
                        else:
                            confusion_matrix.loc[2, 'pred_3'] += 1

            total_count = len(test_df)
            # print(confusion_matrix)
            accuracy += (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[2, 2]) * 100 / total_count
        
            p0 = divide(confusion_matrix.iloc[0, 0] * 100, (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[2, 0]))
            r0 = divide(confusion_matrix.iloc[0, 0] * 100, (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[0, 2]))
            p1 = divide(confusion_matrix.iloc[1, 1] * 100, (confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[2, 1]))
            r1 = divide(confusion_matrix.iloc[1, 1] * 100, (confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2]))
            p2 = divide(confusion_matrix.iloc[2, 2] * 100, (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2] + confusion_matrix.iloc[2, 2]))
            r2 = divide(confusion_matrix.iloc[2, 2] * 100, (confusion_matrix.iloc[2, 0] + confusion_matrix.iloc[2, 1] + confusion_matrix.iloc[2, 2]))

            precision += (p0 + p1 + p2) / 3
            recall += (r0 + r1 + r2) / 3
            f1_score += 2 * (divide((p0 * r0), (p0 + r0)) + divide((p1 * r1) ,(p1 + r1)) + divide((p2 * r2), (p2 + r2))) / 3



            # print(confusion_matrix)
            total_confusion_matrix.iloc[0, 0] += confusion_matrix.iloc[0, 0]
            total_confusion_matrix.iloc[1, 0] += confusion_matrix.iloc[1, 0]
            total_confusion_matrix.iloc[2, 0] += confusion_matrix.iloc[2, 0]
            total_confusion_matrix.iloc[0, 1] += confusion_matrix.iloc[0, 1]
            total_confusion_matrix.iloc[1, 1] += confusion_matrix.iloc[1, 1]
            total_confusion_matrix.iloc[2, 1] += confusion_matrix.iloc[2, 1]
            total_confusion_matrix.iloc[0, 2] += confusion_matrix.iloc[0, 2]
            total_confusion_matrix.iloc[1, 2] += confusion_matrix.iloc[1, 2]
            total_confusion_matrix.iloc[2, 2] += confusion_matrix.iloc[2, 2]

        accuracy /= k_fold
        precision /= k_fold
        recall /= k_fold
        f1_score /= k_fold

        print(f"For Neural network with {layers} hidden layers and {neurons} neurons in each layer:", accuracy, precision, recall, f1_score)
        print(total_confusion_matrix)

