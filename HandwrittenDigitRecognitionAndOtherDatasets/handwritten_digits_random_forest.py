from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random_forest
import pandas as pd

def normalise(train_df, test_df):
    max_values = train_df.max()
    min_values = train_df.min()

    ranges = max_values - min_values
    ranges[ranges == 0] = 1

    train_df = 2 * (train_df - min_values) / ranges - 1
    test_df = 2 * (test_df - min_values) / ranges - 1

    return train_df, test_df

def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def divide (a, b):
    if b == 0:
        return 0
    else:
        return a / b

def k_fold_partition(df, k, i, output):
    # print(df[output])    
    df_i = [None] * 10

    for j in range(10):
        df_i[j] = df[df[output] == j]

    # print(len(df_0), len(df_1))

    # print(df[output].value_counts()) 

    test_df = pd.DataFrame()   

    for j in range(10):
        test_df = pd.concat([test_df, df_i[j][int((len(df_i[j]) * i) / k) : int((len(df_i[j]) * (i + 1)) / k)]])
    
    # print(len(test_df))
    train_df = df.drop(test_df.index)

    return train_df, test_df

digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]
df = pd.DataFrame(digits_dataset_X)
df['Label'] = digits_dataset_y
output_class = 'Label'

k_fold = 10
ntree = [1, 5, 10, 20, 30, 40, 50]
attr = df.columns.to_numpy()
attr = attr[attr != 'Label'].tolist()

for i in range(len(attr)):
    attr[i] = [attr[i], True]

print(attr)
iterations = len(ntree)
y_accuracy = np.zeros(iterations)
y_precision = np.zeros(iterations)
y_recall = np.zeros(iterations)
y_f1 = np.zeros(iterations)
idx=0

idx = 0
for iter in ntree:
    shuffle(df)
    print(iter)
    k_fold = 10
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    total_confusion_matrix = pd.DataFrame(np.zeros((10, 10)), index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'])

    for k in range(k_fold):
        a = 0
        p = [None] * 10
        r = [None] * 10
        f1 = [None] * 10
        train_df, test_df = k_fold_partition(df, k_fold, k, output_class)
        # print("ITERATION " + str(k))
        # print("Length of train_df: ", len(train_df))
        
        rf = random_forest.generate_random_forest(train_df, attr, iter, output_class)
        confusion_matrix = pd.DataFrame(np.zeros((10, 10)), index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'])


        for i in range(len(test_df)):
            output = random_forest.majority_class(rf, test_df.iloc[i])
            expected = test_df[output_class].iloc[i]
            confusion_matrix.iloc[expected, output] += 1
        for i in range(10):
            # print(i, confusion_matrix.iloc[i, i])
            a += confusion_matrix.iloc[i, i]
        accuracy += a / len(test_df)
        for i in range(10):
            p[i] = divide(confusion_matrix.iloc[i, i] * 100, np.sum(confusion_matrix.iloc[:, i]))
            r[i] = divide(confusion_matrix.iloc[i, i] * 100, np.sum(confusion_matrix.iloc[i, :]))
            f1[i] = divide(2 * p[i] * r[i], (p[i] + r[i]))
            # print(p[i], r[i], f1[i])
        precision += np.average(p)
        recall += np.average(r)
        f1_score += np.average(f1)

        # print(confusion_matrix)
        total_confusion_matrix += confusion_matrix
    accuracy /= k_fold
    precision /= k_fold
    recall /= k_fold
    f1_score /= k_fold
    y_accuracy[idx] = accuracy
    y_precision[idx] = precision
    y_recall[idx] = recall
    y_f1[idx] = f1_score
    idx += 1

    print(f"For Random Forest with {iter} trees:", accuracy * 100, precision, recall, f1_score)
    print(total_confusion_matrix)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0,0].plot(ntree, y_accuracy, label='Accuracy')
axs[0,0].set_xlabel('ntree')
axs[0,0].set_ylabel('Accuracy')
axs[0,0].set_title('Accuracies vs ntree')
axs[0,0].grid(visible=True)


axs[0,1].plot(ntree, y_precision, label='Precision')
axs[0,1].set_xlabel('ntree')
axs[0,1].set_ylabel('Precision')
axs[0,1].set_title('Precision vs ntree')
axs[0,1].grid(visible=True)


axs[1,0].plot(ntree, y_recall, label='Recall')
axs[1,0].set_xlabel('ntree')
axs[1,0].set_ylabel('Recall')
axs[1,0].set_title('Recall vs ntree')
axs[1,0].grid(visible=True)


axs[1,1].plot(ntree, y_f1, label='F1 score')
axs[1,1].set_xlabel('ntree')
axs[1,1].set_ylabel('F1 score')
axs[1,1].set_title('F1 score vs ntree')
axs[1,1].grid(visible=True)

plt.tight_layout()
plt.show()

# digit_to_show = np.random.choice(range(N), 1)[0]
# print("Attributes:", digits_dataset_X[digit_to_show])
# print("Class:", digits_dataset_y[digit_to_show])
# plt.imshow(np.reshape(digits_dataset_X[digit_to_show],(8,8)))
# plt.show()
