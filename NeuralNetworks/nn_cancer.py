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

def k_fold_partition(df, k, i, output):
    # print(df[output])    

    df_0 = df[df[output] == 0.0]
    df_1 = df[df[output] == 1.0]

    # print(len(df_0), len(df_1))

    # print(df['class'].value_counts())    

    test_df = pd.concat([df_0[int((len(df_0) * i) / k) : int((len(df_0) * (i + 1)) / k)], 
                         df_1[int((len(df_1) * i) / k) : int((len(df_1) * (i + 1)) / k)]])
    train_df = df.drop(test_df.index)

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
k_fold = 10

r_lambda = 5
alpha = 0.5
output_class = 'Class'
for layers in range(0, 5, 1):
    for neurons in (2 ** x for x in range(1, 5, 1)):
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        total_confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])

        for k in range(k_fold):

            train_df, test_df = k_fold_partition(df, k_fold, k, output_class)
            y_train = train_df[output_class]
            # print(np.unique(y_train, return_counts=True))
            y_train = np.zeros((len(train_df), y_train.nunique()))

            for i in range(len(train_df)):
                # print(train_df[output_class].iloc[i])
                y_train[i][int(train_df[output_class].iloc[i])] = 1
            
            train_df = train_df.drop(columns=[output_class])

            # print(train_df.head())
            nn = neural_network.NeuralNetwork(train_df, layers, [neurons] * layers, r_lambda, alpha, y_train)
            nn.fit()

            confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])
            y_test = test_df[output_class]
            test_df = test_df.drop(columns=output_class)

            # print(train_df.shape)
            # print(test_df.shape)

            for i in range(len(test_df)):
                output = nn.test(test_df.iloc[i])
                # print(output, y_test.iloc[i])
                if output == y_test.iloc[i]:
                    if output == 0:
                        confusion_matrix.loc[0, 'pred_pos'] += 1
                    elif output == 1:
                        confusion_matrix.loc[1, 'pred_neg'] += 1
                else:
                    if output == 0:
                        confusion_matrix.loc[1, 'pred_pos'] += 1
                    elif output == 1:
                        confusion_matrix.loc[0, 'pred_neg'] += 1

            # print(confusion_matrix)
            total_confusion_matrix.iloc[0, 0] += confusion_matrix.iloc[0, 0]
            total_confusion_matrix.iloc[1, 0] += confusion_matrix.iloc[1, 0]
            total_confusion_matrix.iloc[0, 1] += confusion_matrix.iloc[0, 1]
            total_confusion_matrix.iloc[1, 1] += confusion_matrix.iloc[1, 1]

            total_count = confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[0, 1]

            # print(confusion_matrix)
            accuracy += (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1]) * 100 / total_count
            p0 = divide(confusion_matrix.iloc[0, 0] * 100, (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 0]))
            r0 = divide(confusion_matrix.iloc[0, 0] * 100, (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[0, 1]))
            p1 = divide(confusion_matrix.iloc[1, 1] * 100, (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[0, 1]))
            r1 = divide(confusion_matrix.iloc[1, 1] * 100, (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 0]))
            precision += (p0 + p1) / 2
            recall += (r0 + r1) / 2
            f1_score += divide(p0 * r0, (p0 + r0)) + divide(p1 * r1, (p1 + r1))

        accuracy /= k_fold
        precision /= k_fold
        recall /= k_fold
        f1_score /= k_fold

        print(f"For Neural network with {layers} hidden layers and {neurons} neurons in each layer:", accuracy, precision, recall, f1_score)
        print(total_confusion_matrix)

