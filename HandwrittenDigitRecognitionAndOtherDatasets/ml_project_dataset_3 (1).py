# -*- coding: utf-8 -*-
"""ML_project_dataset_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eOMceqm46kNt5LV7Q5DgAbz65YHIguhq
"""

from google.colab import drive
drive.mount('/content/drive/')

import pandas as pd
with open('/content/drive/My Drive/datasets/final_project/loan.csv','r') as file:
  data=pd.read_csv(file) #change csv data into a pandas data frame

data[100:105]
len(data)

from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
data_categorical = data[categorical_cols]
encoder = OneHotEncoder(sparse=False)
encoded_cols = encoder.fit_transform(data_categorical)
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
data_drop = data.drop(columns=categorical_cols)
df_encoded = pd.concat([data_drop, encoded_df], axis=1)

df_encoded[20:21]

def k_folds(X,k):
  class_wise_folds=dict()
  grouped = X.groupby("target")
  # print("grouped", grouped)
  for index , rows in grouped:
    k_x_sets = np.array_split(rows,k)
    class_wise_folds[index] = k_x_sets

  X_array=list()
  for i in range(k):
    fold = list()
    for label in class_wise_folds:
        fold_data = class_wise_folds[label][i]
        if isinstance(fold_data, str):
          fold_data = pd.DataFrame([fold_data])
        fold.append(fold_data)
    X_array.append(pd.concat(fold))
  # print(X_array)
  return X_array

df_encode = df_encoded.rename(columns={"Loan_Status": 'target'})
df_encode.head()

labels = df_encode["target"]
print("labels",labels[0:5])
df_encode = df_encode.drop(columns=["Loan_ID"])
df_encode[0:5]

#knn
from sklearn.model_selection import train_test_split
import random
import numpy as np
import statistics


accuracy_train = list()
f1_train = list()
accuracy_test =list()
f1_test = list()


for k in range(3,18,2):
  accuracy_train_per_k=list()
  f1_score_train_per_k =list()
  accuracy_test_per_k = list()
  f1_score_test_per_k =list()

  k_fold=10
  foldSize = len(df_encode)//k_fold
  folds = k_folds(df_encode,k_fold)

  for iter in range(0,k_fold):
    tp=0
    fp=0
    fn=0
    tn=0
    current_test = folds[iter]
    current_train = pd.DataFrame()
    for j in range(k_fold):
      if(j!=iter):
        current_train = pd.concat([current_train, folds[j]])

    # print(len(X_train))
    # print(len(Y_train))
    # print(current_test)
    # print("hi")
    X_train = current_train.iloc[:, :-1].values
    Y_train = current_train["target"]
    X_test = current_test.iloc[:, :-1].values
    Y_test = current_test["target"]
    X_train = np.delete(X_train, 4,axis=1)
    X_test = np.delete(X_test, 4,axis=1)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    # break
    #normalise
    for column in X_train.columns:
      X_train[column] = ((X_train[column] - X_train[column].min())/(X_train[column].max()-X_train[column].min()))
    # print(Y_train)
    # break
    # print(X_train[0:5])
    # break
    # X_train = X_train[]
    # #Accuracy for train data
    for index1, row1 in X_train.iterrows():
      #calculating euclidean distance
      eucl_distance= list()
      for index2, row2 in X_train.iterrows():
        eucl_dist_row = list()
        # print("row1 = ", row1)
        # print("row2 = ", row2)
        e_d = np.linalg.norm(row1 - row2)
        eucl_distance.append([e_d,labels[index2]])
      eucl_dist_sort = sorted(eucl_distance)
      # print(eucl_distance)
      count_y = sum(1 for item in eucl_dist_sort[0:k] if item[1] == 'Y')
      count_n = sum(1 for item in eucl_dist_sort[0:k] if item[1] == 'N')
      if(count_y>=count_n):
        if('Y' == labels[index1]):
          tp=tp+1
        else:
          fp=fp+1
      else:
        if('N' == labels[index1]):
          tn=tn+1
        else:
          fn=fn+1
    print("tpp_train = ", tp,fp,fn,tn)
    accuracy = (tp+tn)/len(X_train)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    print("accuracy = ", accuracy, "precsion = ", precision)
    accuracy_train_per_k.append(accuracy)
    f1_score_train_per_k.append(f1_score)

    #test
    tp=0
    fp=0
    fn=0
    tn=0
    for index1, row1 in X_test.iterrows():
      #calculating euclidean distance
      eucl_distance= list()
      for index2, row2 in X_train.iterrows():
        eucl_dist_row = list()
        # print("row1 = ", row1)
        # print("row2 = ", row2)
        e_d = np.linalg.norm(row1 - row2)
        eucl_distance.append([e_d,labels[index2]])
      eucl_dist_sort = sorted(eucl_distance)
      # print(eucl_distance)
      count_y = sum(1 for item in eucl_dist_sort[0:k] if item[1] == 'Y')
      count_n = sum(1 for item in eucl_dist_sort[0:k] if item[1] == 'N')
      # print("y and n ", count_y, count_n)
      if(count_y>=count_n):
        if('Y' == labels[index1]):
          tp=tp+1
        else:
          fp=fp+1
      else:
        if('N' == labels[index1]):
          tn=tn+1
        else:
          fn=fn+1
    # print("tpp = ", tp,fp,fn,tn)
    accuracy = (tp+tn)/len(X_test)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    # print("accuracy test= ", accuracy, "precsion test= ", precision)
    accuracy_test_per_k.append(accuracy)
    f1_score_test_per_k.append(f1_score)
  accuracy_train.append(statistics.mean(accuracy_train_per_k))
  accuracy_test.append(statistics.mean(accuracy_test_per_k))
  f1_train.append(statistics.mean(f1_score_train_per_k))
  f1_test.append(statistics.mean(f1_score_test_per_k))

accuracy_train
f1_train

accuracy_test

import matplotlib.pyplot as plt


x_values = list(range(3, 20, 2))

# Plotting each graph
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x_values, accuracy_train, marker='o', color='r')
plt.title('Graph 1')
plt.xlabel('K values')
plt.ylabel('average Accuracy of train')

plt.subplot(2, 2, 2)
plt.plot(x_values, f1_train, marker='o', color='g')
plt.title('Graph 2')
plt.xlabel('K values')
plt.ylabel('average f1 measure of train')

plt.subplot(2, 2, 3)
plt.plot(x_values, accuracy_test, marker='o', color='b')
plt.title('Graph 3')
plt.xlabel('K values')
plt.ylabel('average Accuracy of test')

plt.subplot(2, 2, 4)
plt.plot(x_values, f1_test, marker='o', color='c')
plt.title('Graph 4')
plt.xlabel('K values')
plt.ylabel('average f1_measure of test')

plt.tight_layout()
plt.show()
print(accuracy_train)
print(f1_train)
print(accuracy_test)
print(f1_test)