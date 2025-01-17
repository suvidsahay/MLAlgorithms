# -*- coding: utf-8 -*-
"""ML_project_dataset_3_NN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Q3q0MjL-6bmW0Zg5kpEuOp6n3SyBY0jy
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
  # print("blaaaaaa", X[0:5])
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
df_encode["target"] = df_encode['target'].replace({'N':0,'Y':1})
labels = df_encode["target"]
df_encode = df_encode.drop(columns=["Loan_ID","target"])
df_encode["target"]= labels


df_encode[0:5]

import numpy as np

def sigmoidFunction(z):
  return 1/(1+np.exp((-1)*z))

def forwardPropagationEachLayer(alprev, weights):
  alcurrent = []
  # print("weights", weights)
  # print("alprev", alprev)
  ali=np.dot(weights,alprev)
  # print("aaaaaaa", ali)
  ali = sigmoidFunction(ali)
  # print("ali", ali)
  # exit(0)
  return ali

def forwardPropagation(alprev, label, layeringArray, weights):
  activationArray = []
  activationArray.append(alprev)
  layeringIndex = 1
  while(layeringIndex < len(layeringArray)):
    alprev =np.insert(alprev, 0, 1, axis=0)
    # print("a1111", alprev)
    # print("weights brfore", weights[layeringIndex-1])
    alcurrent = forwardPropagationEachLayer(alprev, weights[layeringIndex-1])
    activationArray.append(alcurrent)
    alprev = alcurrent
    layeringIndex = layeringIndex + 1
    # print("activationArray", activationArray)
  return activationArray

def assign_weights(features_count,layer):
  # print("layerr", layer)
  weights = []
  for i in range(len(layer) -1):
    # print("ufisjd", i)
    rows = layer[i+1]
    # print("rows", rows)
    columns = layer[i] + 1
    # print("columns", columns)
    weight_initial = np.random.uniform(low=-1,high=1, size=(rows,columns))
    weight_initial[weight_initial == 0] = np.random.uniform(low=0, high=1,size=(np.count_nonzero(weight_initial == 0),))
    weights.append(weight_initial)
  return weights

def sumOfSquareOfWeights(weights):
  flat = np.concatenate([np.array(w).flatten() for w in weights])
  squaredSum = np.sum(flat**2)
  return squaredSum

def cost_function(x_train,y_train,layers,weights, lamb):
  layerSize = len(layers)
  total_cost = 0
  # print("HI")
  # print(len(x_train))
  for index in range(len(x_train)):
    # print("heyy")
    x=np.array(x_train[index]).reshape(-1,1)
    y=np.array(y_train[index]).reshape(-1,1)
    # print("\n\nprocessing training instance", index+1)
    # print("Forward processing input", x)
    # print(x)
    # print("y", y)
    activationArray = forwardPropagation(x, y, layers, weights)
    # print(activationArray)
    f_x = activationArray[-1]
    # print("predicted f_x of instance", index+1, f_x)
    # print("actual y of instance", index + 1,y)
    j=0
    if(y == [['Y']]):
      l=1
    else:
      l=0
    for i in range(0,len(y)):
      j=j -l*np.log(f_x[i]) + (l - 1)*np.log(1-f_x[i])
    # print("cost j of instance ", index + 1, j)


    total_cost = total_cost + j
  s= sumOfSquareOfWeights(weights)
  s = (lamb/(2*len(x_train))) * s
  total_cost = total_cost/len(x_train)
  return total_cost + s

def backPropagationGradientcalculator(initial_delta, activationArray, weights, layers, gradient_list):
  layerSize= len(layers)
  delta_list = [None]*layerSize
  delta_list[layerSize - 1] = initial_delta
  for layer in range(layerSize - 2, -1 , -1):
    # print("layer", layer)
    thetak_transpose = np.transpose((weights[layer]))
    # print("weight", weights[layer])
    # print("thetaktranspose", thetak_transpose)
    # print("delta list", delta_list[layer+1])
    # print("activationArray", activationArray)
    activationArray[layer] = np.insert(activationArray[layer], 0, 1, axis=0)
    deltak = np.dot(thetak_transpose, delta_list[layer + 1])*activationArray[layer]*(1 - activationArray[layer])
    delta_list[layer] = deltak[1:]
    delta_list[layer] = delta_list[layer].reshape(-1,1)
    # if(layer!=0):
    #   print("delta ",layer + 1," = ",delta_list[layer])

  for layer in range(layerSize - 2, -1, -1):
    activationArrayTranspose = np.transpose(activationArray[layer])
    # print("activationarraytranspopse",activationArrayTranspose)
    gradient_list[layer] += delta_list[layer+1]*activationArrayTranspose
    # print("gradient of theta", layer + 1, " ", delta_list[layer+1]*activationArrayTranspose)
  return

def backpropagation(x_train,y_train,layers,weights, lamb, alpha,iteration):
  cost=[]
  for inn in range(0,iteration,1):
    # print(inn, "aayoo")
    layerSize = len(layers)
    gradient_list=[]
    for i in range(layerSize - 1):
      gradient_list.append(np.zeros((layers[i+1], layers[i] + 1)))
    total_cost = 0
    for index in range(len(x_train)):
      x=np.array(x_train[index]).reshape(-1,1)
      y=np.array(y_train[index]).reshape(-1,1)
      # print(y)
      if(y == [[0]]):
        y = np.array([[1], [0]])
      else:
        y = np.array([[0], [1]])

      # print("Computing gradient for instance", index+1)
      # print(x)
      activationArray = forwardPropagation(x, y, layers, weights)
      # print(activationArray)
      f_x = activationArray[-1]
      # print("predicted f_x of instance", index+1, f_x)
      # print("actual y of instance", index + 1,y)
      # print("delta", len(layers)," = ", f_x -y)
      backPropagationGradientcalculator(f_x - y, activationArray, weights, layers, gradient_list)

    # print("\nThe entire training set has been processed. Compuing the average regularised gradient\n")
    for layer in range(layerSize - 2, -1, -1):
      # print(weights[layer])
      # print(lamb)
      pk = lamb*weights[layer]
      # print(pk)
      # pk[:,0]=0
      gradient_list[layer]= gradient_list[layer] + pk
      gradient_list[layer]= gradient_list[layer]/len(x_train)
      # print("reg gradient of theta", layer + 1, " " , gradient_list[layer])

    for layer in range(layerSize-2,-1,-1):
      weights[layer] = weights[layer] - alpha*gradient_list[layer]
    total_cost_train = cost_function(x_train,y_train,layers,weights, 0.250)
    # print("total cost j", total_cost_train)
    cost.append(total_cost_train)
  return weights,cost

def evaluate(actual_list,predicted_list):
  print(actual_list, predicted_list)
  tp=0
  tn=0
  fp=0
  fn=0
  for i in range(len(actual_list)):
    if(actual_list[i] == predicted_list[i]):
      if(actual_list[i]==0):
        tp=tp+1
      else:
        tn=tn+1
    else:
      if(actual_list[i]==0):
        fn=fn+1
      else:
        fp=fp+1
  accuracy = (tp + tn)/len(actual_list)
  if(tp + fp == 0 or tp==0):
    precision =1
  else:
    precision = tp/(tp+fp)
  if(tp + fn == 0 or tp == 0):
    recall = 1
  else:
    recall = tp/(tp+fn)
  f1_measure = 2*(precision * recall)/ (precision + recall)
  return accuracy,f1_measure

def predict(X,Y, layers, weights):
  predicted_list = []
  actual_list = []

  for i in range(len(X)):
    x_instance = np.array(X[i]).reshape(-1,1)
    y_instance = Y[i]
    activationArray = forwardPropagation(x_instance,y_instance,layers,weights)
    f_x = activationArray[-1]
    index = np.argmax(f_x)
    actual_list.append(y_instance)
    predicted_list.append(index)
  accuracy, f1_measure = evaluate(actual_list,predicted_list)
  return accuracy,f1_measure

#nn
from sklearn.model_selection import train_test_split
import random
import numpy as np
import statistics


accuracy_train = list()
f1_train = list()
accuracy_test =list()
f1_test = list()
k_fold=10
foldSize = len(df_encode)//k_fold
folds = k_folds(df_encode,k_fold)
hidden_layers = [[21,16,2],[21,32,2]]


accuracy=[]
f1_measure=[]
for layers in hidden_layers:
  for i in range(0,k_fold):
    # print("fold no", i)
    # print("layers", layers)
    # print("folddd", folds[i])
    current_test = folds[i]
    current_train = pd.DataFrame()
    for j in range(k_fold):
      if(j!=i):
        current_train = pd.concat([current_train, folds[j]])


    X_train = current_train.iloc[:, :-1]
    for column in X_train.columns:
      X_train[column] = ((X_train[column] - X_train[column].min())/(X_train[column].max()-X_train[column].min()))
    Y_train = current_train.iloc[:, -1]

    X_test = current_test.iloc[:, :-1]
    Y_test = current_test.iloc[:, -1]
    for column in X_test.columns:
      X_test[column] = ((X_test[column] - X_test[column].min())/(X_test[column].max()-X_test[column].min()))
    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)
    # print(len(X_train.columns))
    # print(np.array(X_train))
    # print("helllo",X_train[0:5].values)
    # break
    # print("y_train", Y_train[0])
    features_count = X_train.shape[1]
    # print(Y_train)

    weights = assign_weights(features_count, layers)
    # print(weights)
    total_cost_train = cost_function(X_train.values,Y_train.values,layers,weights, 0.250)
    # print("total cost j", total_cost_train)
    # # print("\n\n-----------------------------------------------")
    # print("Running backpropagation")
    weights, cost = backpropagation(X_train.values,Y_train.values,layers,weights,0.2, 0.1, 500)

    accuracy_each, f1_measure_each = predict(X_test.values, Y_test.values, layers, weights)
    accuracy.append(accuracy_each)
    f1_measure.append(f1_measure_each)

cost

import matplotlib.pyplot as plt

values=cost

# Extracting the values
y_values = [value[0] for value in values]

# X values (assuming the x-axis starts from 3 and increments by 2)
x_values = range(len(values))  # Change 48 to the maximum value of x if needed

# Plotting
plt.plot(x_values, y_values, marker='o')

# Adding labels and title
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Graph of Y Values over X Values')

# Display the plot
plt.show()

fig, ax = plt.subplots()

ax.plot(x_values, y_values, label='Cost')

ax.set_title('Cost function vs number of training samples')
ax.set_xlabel('Number of training sample')
ax.set_ylabel('Cost')

ax.legend()

plt.grid(True)
plt.show()

accuracy_1 = sum(accuracy[:10])/10
accuracy_2 = sum(accuracy[10:])/10
f1_measure_1 = sum(f1_measure[:10])/10
f1_measure_2 = sum(f1_measure[10:])/10

print(accuracy_1)
print(accuracy_2)
print(f1_measure_1)
print(f1_measure_2)

import matplotlib.pyplot as plt

# Define a range of training set sizes (e.g., from 10% to 100% of the total dataset)
training_set_sizes = [int(len(df_encode) * pct) for pct in np.linspace(0.1, 1.0, 10)]

# Initialize lists to store performance metrics

# Define neural network parameters and architecture
layers = [[21, 16, 2],[21,32,2]]
alpha = 0.1  # Step size
costs_layer=[]
# For each training set size
for hidden_layers in layers:
  test_costs = []
  for train_size in training_set_sizes:
      # Split the data into training and test sets
      train_data = df_encode.iloc[:train_size]
      test_data = df_encode.iloc[train_size:]

      # Check if the test set is empty
      if len(test_data) == 0:
          continue

      X_train = train_data.iloc[:, :-1]
      Y_train = train_data.iloc[:, -1]
      X_test = test_data.iloc[:, :-1]
      Y_test = test_data.iloc[:, -1]

      # Normalize the features
      for column in X_train.columns:
          X_train[column] = (X_train[column] - X_train[column].min()) / (X_train[column].max() - X_train[column].min())
          X_test[column] = (X_test[column] - X_test[column].min()) / (X_test[column].max() - X_test[column].min())

      # Initialize weights
      features_count = X_train.shape[1]
      weights = assign_weights(features_count, hidden_layers)

      # Train the neural network model
      backpropagation(X_train.values, Y_train.values, hidden_layers, weights, 1, alpha, 500)

      # Evaluate the model's performance on the test set
      total_cost_test = cost_function(X_test.values, Y_test.values, hidden_layers, weights, 1)
      test_costs.append(total_cost_test)
  costs_layer.append(test_costs)

# print(test_costs)
# Plot the learning curve
plt.plot(training_set_sizes[:len(costs_layer[0])], costs_layer[0], marker='o')
plt.xlabel('Number of training examples')
plt.ylabel('Test set cost')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

print("Step size (alpha):", alpha)

plt.plot(training_set_sizes[:len(costs_layer[1])], costs_layer[1], marker='o')
plt.xlabel('Number of training examples')
plt.ylabel('Test set cost')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

print("Step size (alpha):", alpha)

