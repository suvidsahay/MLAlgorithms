import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import random
import math

def bootstrap(train_df):
    n = len(train_df)
    train_df_copy = train_df.copy().reset_index(drop=True)
    # print(train_df_copy)

    train_df_bootstrapped = train_df_copy.sample(n, replace = True)

    return train_df_bootstrapped


def entropy(dataset, target_attr):
    entropy = 0
    counts = dataset[target_attr].value_counts()
    sum = counts.sum()

    for i in range(0, 4):
        if counts.get(i, 0) != 0:
            entropy += (counts.get(i, 0) / sum) * np.log2(counts.get(i, 0) / sum) * -1
    return entropy

def gini(dataset, target_attr):
    gini = 0
    counts = dataset[target_attr].value_counts()
    sum = counts.sum()

    for i in range(0, 4):
        if counts.get(i, 0) != 0:
            gini += np.square(counts.get(i, 0) / sum)
    gini = 1-gini
    return gini

def calc_info(dataset, attribute, target_attr):

    dataset_copy = dataset.copy()

    info = 0
    total_size = len(dataset_copy)
    for i in range(3):
        temp_dataset = dataset_copy[dataset_copy[attribute] == i]
        partition_size = len(temp_dataset)
        info += (partition_size * entropy(temp_dataset, target_attr)) / total_size

    return info

def calc_numerical_info(dataset, attribute, target_attr):

    dataset_copy = dataset.copy()

    total_size = len(dataset_copy)
    threshold = np.mean(dataset[attribute])

    temp_dataset = dataset_copy[dataset_copy[attribute] < threshold]
    partition_size = len(temp_dataset)
    info = (partition_size * entropy(temp_dataset, target_attr)) / total_size
    temp_dataset = dataset_copy[dataset_copy[attribute] >= threshold]
    partition_size = len(temp_dataset)
    info += (partition_size * entropy(temp_dataset, target_attr)) / total_size

    return info, threshold


def random_attributes(attributes):
    m = math.ceil(np.sqrt(len(attributes)))
    random_attributes = []

    for i in range(m):
        random_attributes.append(random.choice(attributes))

    return random_attributes

def majority_class(random_forest, test_dataset):
    class_freq = defaultdict(int)
    test_dataset_copy = test_dataset.copy().reset_index(drop=True)
    # print(test_dataset)
    for tree in random_forest:
        output = test_tree(tree, test_dataset)
        class_freq[output] += 1
        # print(tree)
        # print(output)
    # print(test_dataset)
    # print(class_freq, max(class_freq, key = class_freq.get), test_dataset['class'])
    
    return max(class_freq, key = class_freq.get)

class Node:
    def __init__(self):
        self.c0 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.attribute = None #attibute being tested
        self.is_numerical = False
        self.threshold = None #Threshold for splitting the nodes on numerical attributes
        self.is_leaf = False
        self.label = None #label for leaf nodes

    def __str__(self, level=0, value=None):
        if self.is_leaf:
            return "\t"*(level)+value+"\t"+str(self.label)+"\n"
        ret = "\t"*level+str(value)+"\t"+self.attribute+"\n"
        
        if self.is_numerical:
            if self.c0 != None:
                ret += self.c0.__str__(level+1, 'c0 ' + f'{self.threshold:.1f}')
            if self.c1 != None:
                ret += self.c1.__str__(level+1, 'c1 ' + f'{self.threshold:.1f}')
        else:
            if self.c0 != None:
                ret += self.c0.__str__(level+1, 'c0')
            if self.c1 != None:
                ret += self.c1.__str__(level+1, 'c1')
            if self.c2 != None:
                ret += self.c2.__str__(level+1, 'c2')
            if self.c3 != None:
                ret += self.c3.__str__(level+1, 'c3')
            if self.c4 != None:
                ret += self.c4.__str__(level+1, 'c4')
        return ret

def create_tree(dataset, attributes, target_class, depth = 0):

    #Already classified branch
    # print(dataset)
    if len(dataset[target_class].unique()) == 1:
        node = Node()
        node.is_leaf = True
        node.label = dataset[target_class].iloc[0]
        return node
    
    # # Stop if the majority class is greater than 85%
    # total_count = len(dataset)
    # counts = dataset['class'].value_counts()
    # target_max_count = counts.idxmax()
    # max_count = counts[target_max_count]

    # if max_count * 100 > total_count * 85:
    #     node = Node()
    #     node.is_leaf = True
    #     node.label = target_max_count
    #     return node

    # Stop if there are no more than n data in the current dataset
    if len(dataset) <= 5:
        label = dataset[target_class].value_counts().idxmax()
        node = Node()
        node.is_leaf = True
        node.label = label
        return node

    #No more attributes left
    if not attributes or depth >= 3:
        # print(dataset)
        label = dataset[target_class].value_counts().idxmax()
        # print(dataset['class'].value_counts())
        # print(label)

        node = Node()
        node.is_leaf = True
        node.label = label
        return node

    entropy_current = entropy(dataset, target_class)
    gain_list = list()
    # gini_current = gini(dataset, target_class)
    # gini_list = list()

    # for attribute, is_numerical in random_attributes(attributes):
    for attribute, is_numerical in random_attributes(attributes):

        info = 0
        threshold = None
        if is_numerical:
            info, threshold = calc_numerical_info(dataset, attribute, target_class)
        else:
            info = calc_info(dataset, attribute, target_class)
        gain_list.append(((attribute, is_numerical, threshold), entropy_current - info))
        # gini_list.append(((attribute, is_numerical, threshold), info))

    gain_list = sorted(gain_list, key=lambda a: a[1], reverse=True)
    # gini_list = sorted(gini_list, key=lambda a: a[1])

    attr = gain_list[0][0][0]
    # print(dataset)
    # print(attributes)
    # print(gain_list)
    # print(attr)

    node = Node()
    node.is_leaf = False
    node.attribute = attr
    if gain_list[0][0][1]:
        node.is_numerical = True
        node.threshold = gain_list[0][0][2]
    else:
        node.is_numerical = False

    #temp node if any split is empty
    temp = Node()
    temp.is_leaf = True
    temp.label = dataset[target_class].value_counts().idxmax()

    if gain_list[0][0][1]:
        threshold = gain_list[0][0][2]
        if len(dataset[dataset[attr] >= threshold]) == 0:
            node.c0 = temp
        else:
            node.c0 = create_tree(dataset[dataset[attr] >= threshold], attributes, target_class, depth + 1)

        if len(dataset[dataset[attr] < threshold]) == 0:
            node.c1 = temp
        else:
            node.c1 = create_tree(dataset[dataset[attr] < threshold], attributes, target_class, depth + 1)
    else:
        if len(dataset[dataset[attr] == 0]) == 0:
            node.c0 = temp
        else:
            node.c0 = create_tree(dataset[dataset[attr] == 0], attributes, target_class, depth + 1)
        if len(dataset[dataset[attr] == 1]) == 0:
            node.c1 = temp
        else:
            node.c1 = create_tree(dataset[dataset[attr] == 1], attributes, target_class, depth + 1)
        if len(dataset[dataset[attr] == 2]) == 0:
            node.c2 = temp
        else:
            node.c2 = create_tree(dataset[dataset[attr] == 2], attributes, target_class, depth + 1)
        if len(dataset[dataset[attr] == 3]) == 0:
            node.c3 = temp
        else:
            node.c3 = create_tree(dataset[dataset[attr] == 3], attributes, target_class, depth + 1)
        if len(dataset[dataset[attr] == 4]) == 0:
            node.c4 = temp
        else:
            node.c4 = create_tree(dataset[dataset[attr] == 4], attributes, target_class, depth + 1)
    return node;

def test_tree(node, data):
    if node.is_leaf:
        return node.label

    if node.is_numerical:
        if data[node.attribute] >= node.threshold:
            return test_tree(node.c0, data)
        else:
            return test_tree(node.c1, data)
    else:
        if data[node.attribute] == 0:
            return test_tree(node.c0, data)
        elif data[node.attribute] == 1:
            return test_tree(node.c1, data)
        elif data[node.attribute] == 2:
            return test_tree(node.c2, data)
        elif data[node.attribute] == 3:
            return test_tree(node.c3, data)
        elif data[node.attribute] == 4:
            return test_tree(node.c4, data)
        
def generate_random_forest(train_df, attributes, ntrees, target_class):
    random_forest = list()
    for _ in range(ntrees):
        train_df_copy = bootstrap(train_df)
        root = create_tree(train_df_copy, attributes, target_class)
        random_forest.append(root)
        # print(train_df_copy.index.value_counts())
        # print(root)
    return random_forest
