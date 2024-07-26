import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def k_fold_partition(df, k, i):
    df_0 = df[df['class'] == 0]
    df_1 = df[df['class'] == 1]

    # print(len(df_0), len(df_1))

    # print(df['class'].value_counts())    

    test_df = pd.concat([df_0[int((len(df_0) * i) / k) : int((len(df_0) * (i + 1)) / k)], 
                         df_1[int((len(df_1) * i) / k) : int((len(df_1) * (i + 1)) / k)]])
    train_df = df.drop(test_df.index)

    return train_df, test_df

df = pd.read_csv("datasets/hw3_house_votes_84.csv")

attr = [
    ('handicapped-infants',False),
    ('water-project-cost-sharing', False),
    ('adoption-of-the-budget-resolution', False),
    ('physician-fee-freeze', False),
    ('el-salvador-adi', False),
    ('religious-groups-in-schools', False),
    ('anti-satellite-test-ban', False),
    ('aid-to-nicaraguan-contras', False),
    ('mx-missile', False),
    ('immigration', False),
    ('synfuels-corporation-cutback', False),
    ('education-spending', False),
    ('superfund-right-to-sue', False),
    ('crime', False),
    ('duty-free-exports', False),
    ('export-administration-act-south-africa', False)
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def g(matrix):
    res = [None] * (len(matrix))
    for i in range(len(matrix)):
        res[i] = sigmoid(matrix[i])
    return res
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def g(matrix):
    res = [None] * (len(matrix))
    for i in range(len(matrix)):
        res[i] = sigmoid(matrix[i])
    return res


def cost(A, y):
    return -1 * np.sum(np.multiply(y, np.log(A)) + np.multiply((1 - y), np.log(1 + np.multiply(-1, A))))


def regularisation(T, r_lambda):
    return (sum(np.sum(np.square(matrix[:, 1:])) for matrix in T) * r_lambda) / 2


def train_nn(train_df, layers, neurons, r_lambda, alpha, y):
    A = [None] * (layers + 2)
    T = [None] * (layers + 1)
    T[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    T[1] = np.array([[0.70000, 0.50000, 0.60000]])
    delta = [None] * (layers + 2)
    D = [None] * (layers + 1)
    D[0] = np.zeros((2, 2))
    
    for i in range(layers - 1):
        D[i + 1] = np.zeros((neurons[i + 1], neurons[i] + 1))
    D[layers] = np.zeros((len(y[0]), neurons[layers - 1] + 1))
    
    J = 0
    for i in range(len(train_df)):
        print("\nFor instance: ", i + 1)
        #forward propagation
        x = train_df[i]
        A[0] = np.append([1], x)
        print("A", 0, A[0])
        for j in range(layers):
            A[j + 1] = np.append([1], g(np.matmul(T[j], A[j])))
            print("A", j + 1, A[j + 1])
        A[layers + 1] = g(np.matmul(T[layers], A[layers]))
        print("A", layers + 1, A[layers + 1])
        print("Expected output:", A[layers + 1])
        print("Actual output:", y[i])
        print("Cost associated:", cost(A[layers + 1], y[i]))
        J += cost(A[layers + 1], y[i])
        
        #bakward propagation
        delta[layers + 1] = A[layers + 1] - y[i]
        print("Delta", layers + 1, delta[layers + 1])

        for j in range(layers, 0, -1):
            # print(i, j, T[j].shape, delta[j + 1].shape, A[j].shape)
            delta[j] = np.multiply(np.matmul(np.transpose(T[j]), delta[j + 1]), np.multiply(A[j], 1 - A[j]))
            delta[j] = np.delete(delta[j], 0)
            print("Delta", j, delta[j])

            # print(delta[j].shape, np.transpose(delta[j + 1][np.newaxis]).shape, A[j][np.newaxis].shape)
            D[j] += np.matmul(np.transpose(delta[j + 1][np.newaxis]), A[j][np.newaxis])
            print("Gradient", j, np.matmul(np.transpose(delta[j + 1][np.newaxis]), A[j][np.newaxis]))
        D[0] += np.matmul(np.transpose(delta[1][np.newaxis]), A[0][np.newaxis])
        print("Gradient", 0, np.matmul(np.transpose(delta[1][np.newaxis]), A[0][np.newaxis]))
    
    print("\nThe entire training set has been processes. Computing the average (regularized) gradients:")
    for j in range(layers, -1, -1):
        D[j] = (D[j] + np.multiply(r_lambda, T[j])) / len(train_df)
        T[j] -= np.multiply(alpha, D[j])
        print("Gradient", j, D[j])
    
    J += regularisation(T, r_lambda)
    J /= 2
    print("\nFinal (regularized) cost, J, based on the complete training set: ", J)


train_df = np.array([0.13, 0.42])
y = np.array([[0.9], [0.23]])
train_nn(train_df, 1, [2], 0, 0, y)