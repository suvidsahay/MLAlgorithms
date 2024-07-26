import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt


df = pd.read_csv("iris.csv", names = ["length_sepal", "width_sepal", "length_petal", "width_petal", "species"])


def shuffle_split_and_normalise(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index()
    #split into train and test df
    train_df = df.sample(frac = 0.8)
    test_df = df.drop(train_df.index)

    max_values = train_df.max()
    min_values = train_df.min()

    train_df['length_sepal'] = 2 * (train_df['length_sepal'] - min_values['length_sepal']) / (max_values['length_sepal'] - min_values['length_sepal']) - 1
    train_df['width_sepal'] = 2 * (train_df['width_sepal'] - min_values['width_sepal']) / (max_values['width_sepal'] - min_values['width_sepal']) - 1
    train_df['length_petal'] = 2 * (train_df['length_petal'] - min_values['length_petal']) / (max_values['length_petal'] - min_values['length_petal']) - 1
    train_df['width_petal'] = 2 * (train_df['width_petal'] - min_values['width_petal']) / (max_values['width_petal'] - min_values['width_petal']) - 1

    test_df['length_sepal'] = 2 * (test_df['length_sepal'] - min_values['length_sepal']) / (max_values['length_sepal'] - min_values['length_sepal']) - 1
    test_df['width_sepal'] = 2 * (test_df['width_sepal'] - min_values['width_sepal']) / (max_values['width_sepal'] - min_values['width_sepal']) - 1
    test_df['length_petal'] = 2 * (test_df['length_petal'] - min_values['length_petal']) / (max_values['length_petal'] - min_values['length_petal']) - 1
    test_df['width_petal'] = 2 * (test_df['width_petal'] - min_values['width_petal']) / (max_values['width_petal'] - min_values['width_petal']) - 1

    return train_df, test_df

# Normalise the variables
def normalise(train_df, test_df):
    return train_df, test_df


def run_knn(train_df, test_dataset, k):
    columns = ['length_sepal', 'width_sepal', 'length_petal', 'width_petal']

    train_df_copy = train_df.copy()


    
    #Calculate distance of test_dataset from train_df
    train_df_copy['distance'] = np.sqrt((train_df_copy['length_sepal'] - test_dataset['length_sepal'])**2 +
                                        (train_df_copy['width_sepal'] - test_dataset['width_sepal'])**2 +
                                        (train_df_copy['length_petal'] - test_dataset['length_petal'])**2 +
                                        (train_df_copy['width_petal'] - test_dataset['width_petal'])**2)


    # train_df_copy['distance'] = distance


    train_df_copy.sort_values(inplace=True, by=['distance'], ignore_index=True)

    #freq for calculating maximum occuring label and average distance for tie breaker
    distance_dict_freq = defaultdict(int)
    distance_dict_average = defaultdict(float)


    for i in range(k):

        distance_dict_freq[train_df_copy['species'].iloc[i]] += 1
        distance_dict_average[train_df_copy['species'].iloc[i]] += train_df_copy['distance'].iloc[i]

    distance_list = list()
    for key, value in distance_dict_average.items():
        distance_dict_average[key] = value / distance_dict_freq[key]
        distance_list.append(tuple([key, distance_dict_freq[key], distance_dict_average[key]]))
    
    #sort the labels based on freq and average distance
        
    distance_list = sorted(distance_list, key=lambda a: a[2])
    distance_list = sorted(distance_list, key=lambda a: a[1], reverse=True)

    return distance_list[0][0]
    

iteration = 20

x_y_train = np.zeros((2, iteration, 26))
x_y_test = np.zeros((2, iteration, 26))
for iter in range(iteration):
    train_df, test_df = shuffle_split_and_normalise(df)

    for k in range(1, 52):
        if(k % 2 == 0):
            continue
        x_y_train[0,iter,int(k / 2)] = k
        x_y_test[0,iter,int(k / 2)] = k

        correct_count = 0
        for i in range(len(train_df)):
            label = run_knn(train_df=train_df, test_dataset=train_df.iloc[i], k = k)
            if label == train_df['species'].iloc[i]:
                correct_count += 1
        x_y_train[1,iter,int(k / 2)] = (correct_count * 100) / len(train_df)

        correct_count = 0

        for i in range(len(test_df)):
            label = run_knn(train_df=train_df, test_dataset=test_df.iloc[i], k = k)
            if label == test_df['species'].iloc[i]:
                correct_count += 1
        x_y_test[1,iter,int(k / 2)] = (correct_count * 100) / len(test_df)

y_train_avg = np.mean(x_y_train[1, :, :], axis = 0)
y_test_avg = np.mean(x_y_test[1, :, :], axis = 0)
x_train_avg = np.mean(x_y_train[0, :, :], axis = 0)
x_test_avg = np.mean(x_y_test[0, :, :], axis = 0)
std_dev_train = np.std(x_y_train[1, :, :], axis = 0)
std_dev_test = np.std(x_y_test[1, :, :], axis = 0)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(x_train_avg, y_train_avg, label='Train Accuracy')
axs[0].errorbar(x_train_avg, y_train_avg, std_dev_train, ecolor='red', marker='^', capsize=3)
axs[0].set_xlabel('k')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Training accuracies')

axs[1].plot(x_test_avg, y_test_avg, label='Test Accuracy')
axs[1].errorbar(x_test_avg, y_test_avg, std_dev_test, ecolor='red', marker='^', capsize=3)
axs[1].set_xlabel('k')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Testing accuracies')


plt.show()