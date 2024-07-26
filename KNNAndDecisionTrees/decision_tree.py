import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt

df = pd.read_csv("house_votes_84.csv")

def shuffle_and_split(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index()
    
    #split into train and test df
    train_df = df.sample(frac = 0.8)
    test_df = df.drop(train_df.index)

    return train_df, test_df

def entropy(dataset):
    entropy = 0
    counts = dataset['target'].value_counts()
    sum = counts.sum()

    for i in range(2):
        if counts.get(i, 0) != 0:
            entropy += (counts.get(i, 0) / sum) * np.log2(counts.get(i, 0) / sum) * -1
    return entropy

def gini(dataset):
    gini = 0
    counts = dataset['target'].value_counts()
    sum = counts.sum()

    for i in range(2):
        if counts.get(i, 0) != 0:
            gini += np.square(counts.get(i, 0) / sum)
    gini = 1-gini
    return gini

def calc_info(dataset, attribute):

    dataset_copy = dataset.copy()

    info = 0
    total_size = len(dataset_copy)
    for i in range(3):
        temp_dataset = dataset_copy[dataset_copy[attribute] == i]
        partition_size = len(temp_dataset)
        info += (partition_size * entropy(temp_dataset)) / total_size

    return info



class Node:
    def __init__(self):
        self.y = None
        self.n = None
        self.m = None
        self.attribute = None #attibute being tested
        self.is_leaf = False
        self.label = None #label for leaf nodes

    def __str__(self, level=0, value=None):
        if self.is_leaf:
            return "\t"*(level)+value+"\t"+str(self.label)+"\n"
        ret = "\t"*level+str(value)+"\t"+self.attribute+"\n"
        
        if self.m != None:
            ret += self.m.__str__(level+1, 'm')
        if self.n != None:
            ret += self.n.__str__(level+1, 'n')
        if self.y != None:
            ret += self.y.__str__(level+1, 'y')
        return ret


def create_tree(dataset, attributes):
    #Already classified branch
    
    if len(dataset['target'].unique()) == 1:
        node = Node()
        node.is_leaf = True
        node.label = dataset['target'].iloc[0]
        return node
    
    # # Stop if the majority class is greater than 85%
    # total_count = len(dataset)
    # counts = dataset['target'].value_counts()
    # target_max_count = counts.idxmax()
    # max_count = counts[target_max_count]

    # if max_count * 100 > total_count * 85:
    #     node = Node()
    #     node.is_leaf = True
    #     node.label = target_max_count
    #     return node


    #No more attributes left
    if not attributes:
        label = dataset['target'].value_counts().argmax()
        node = Node()
        node.is_leaf = True
        node.label = label
        return node

    entropy_current = entropy(dataset)
    gain_list = list()
    # gini_current = gini(dataset)
    # gini_list = list()

    for attribute in attributes:
        info = calc_info(dataset, attribute)
        gain_list.append((attribute, entropy_current - info))
        # gini_list.append((attribute, info))


    gain_list = sorted(gain_list, key=lambda a: a[1], reverse=True)
    # gini_list = sorted(gini_list, key=lambda a: a[1])

    attr = gain_list[0][0]
    attributes_copy = attributes.copy()
    attributes_copy.remove(gain_list[0][0])

    node = Node()
    node.is_leaf = False
    node.attribute = attr

    #temp node if any split is empty
    temp = Node()
    temp.is_leaf = True
    temp.label = dataset['target'].value_counts().argmax()

    if len(dataset[dataset[attr] == 2]) == 0:
        node.y = temp
    else:
        node.y = create_tree(dataset[dataset[attr] == 2].drop(attr, axis = 1), attributes_copy)

    if len(dataset[dataset[attr] == 1]) == 0:
        node.n = temp
    else:
        node.n = create_tree(dataset[dataset[attr] == 1].drop(attr, axis = 1), attributes_copy)
    
    if len(dataset[dataset[attr] == 0]) == 0:
        node.m = temp
    else:
        node.m = create_tree(dataset[dataset[attr] == 0].drop(attr, axis = 1), attributes_copy)

    return node;


attr = ['handicapped-infants',
        'water-project-cost-sharing',
        'adoption-of-the-budget-resolution',
        'physician-fee-freeze',
        'el-salvador-adi',
        'religious-groups-in-schools',
        'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras',
        'mx-missile',
        'immigration',
        'synfuels-corporation-cutback',
        'education-spending',
        'superfund-right-to-sue',
        'crime','duty-free-exports',
        'export-administration-act-south-africa']

def test_tree(node, data):
    if node.is_leaf:
        return node.label

    if data[node.attribute] == 0:
        return test_tree(node.m, data)
    elif data[node.attribute] == 1:
        return test_tree(node.n, data)
    else:
        return test_tree(node.y, data)

iterations = 100
y_train = np.zeros(iterations)
y_test = np.zeros(iterations)
for iter in range(iterations):

    train_df, test_df = shuffle_and_split(df)

    root = create_tree(train_df, attr)

    correct_count = 0
    for i in range(len(train_df)): 
        output = test_tree(root, train_df.iloc[i])
        if output == train_df['target'].iloc[i]:
            correct_count+=1
    
    y_train[iter] = (correct_count * 100) / len(train_df)

    correct_count = 0
    for i in range(len(test_df)): 
        output = test_tree(root, test_df.iloc[i])
        if output == test_df['target'].iloc[i]:
            correct_count+=1
    y_test[iter] = (correct_count * 100) / len(test_df)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].hist(y_train, color='blue', bins=40, zorder=3)
axs[0].grid(visible=True)
axs[0].set_xlabel('Accuracy')
axs[0].set_ylabel('Accuracy Frequency')
axs[0].set_title('Training accuracy')
axs[0].text(0.05, 0.95, f'Mean: {np.mean(y_train):.2f}\nStd: {np.std(y_train):.2f}', transform=axs[0].transAxes, verticalalignment='top')

axs[1].hist(y_test, color='blue', bins=40, zorder=3)
axs[1].set_xlabel('Accuracy')
axs[1].grid(visible=True)
axs[1].set_ylabel('Accuracy Frequency')
axs[1].set_title('Testing accuracy')
axs[1].text(0.05, 0.95, f'Mean: {np.mean(y_test):.2f}\nStd: {np.std(y_test):.2f}', transform=axs[1].transAxes, verticalalignment='top')


plt.tight_layout()
plt.show()





    








