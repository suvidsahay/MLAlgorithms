import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
import random_forest

def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def k_fold_partition(df, k, i):
    df_1 = df[df['contraceptive-method-used'] == 1]
    df_2 = df[df['contraceptive-method-used'] == 2]
    df_3 = df[df['contraceptive-method-used'] == 3]

    # print(df['contraceptive-method-used'].value_counts())    

    test_df = pd.concat([df_3[int((len(df_3) * i) / k) : int((len(df_3) * (i + 1)) / k)], 
                         df_1[int((len(df_1) * i) / k) : int((len(df_1) * (i + 1)) / k)],
                         df_2[int((len(df_2) * i) / k) : int((len(df_2) * (i + 1)) / k)]])
    train_df = df.drop(test_df.index)

    return train_df, test_df

headers = ['wife-age', 'wife-education', 'husband-education', 'number-of-children-ever-born', 'wife_religion', 'wife-now-working', 'husband-occupation', 'standard-of-living-index', 'media-exposure', 'contraceptive-method-used']
df = pd.read_csv("datasets/cmc.data", header=None, names=headers)

attr = [
    ('wife-age',True),
    ('wife-education', False),
    ('husband-education', False),
    ('number-of-children-ever-born', True),
    ('wife_religion', False),
    ('wife-now-working', False),
    ('husband-occupation', False),
    ('standard-of-living-index', False),
    ('media-exposure', False)
]

ntree = [1, 5, 10, 20, 30, 40, 50]
# ntree = [1] 

iterations = len(ntree)
y_accuracy = np.zeros(iterations)
y_precision = np.zeros(iterations)
y_recall = np.zeros(iterations)
y_f1 = np.zeros(iterations)

print(df['contraceptive-method-used'].value_counts())
idx = 0
for iter in ntree:
    print(iter)
    df = shuffle(df)
    k_fold = 10
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    for k in range(k_fold):
        train_df, test_df = k_fold_partition(df, k_fold, k)
        
        rf = random_forest.generate_random_forest(train_df, attr, iter, 'contraceptive-method-used')

        confusion_matrix = pd.DataFrame(np.zeros((3, 3)), index=[1, 2, 3], columns=['pred_1', 'pred_2', 'pred_3'])

        for i in range(len(test_df)):
            output = random_forest.majority_class(rf, test_df.iloc[i])
            # print(test_df.iloc[i], output)
            if output == test_df['contraceptive-method-used'].iloc[i]:
                if output == 1:
                    confusion_matrix.loc[1, 'pred_1'] += 1
                elif output == 2:
                    confusion_matrix.loc[2, 'pred_2'] += 1
                else:
                    confusion_matrix.loc[3, 'pred_3'] += 1
            else:
                if output == 1:
                    if test_df['contraceptive-method-used'].iloc[i] == 2:
                        confusion_matrix.loc[2, 'pred_1'] += 1
                    else:
                        confusion_matrix.loc[3, 'pred_1'] += 1
                elif output == 2:
                    if test_df['contraceptive-method-used'].iloc[i] == 1:
                        confusion_matrix.loc[1, 'pred_2'] += 1
                    else:
                        confusion_matrix.loc[3, 'pred_2'] += 1
                elif output == 3:
                    if test_df['contraceptive-method-used'].iloc[i] == 1:
                        confusion_matrix.loc[1, 'pred_3'] += 1
                    else:
                        confusion_matrix.loc[2, 'pred_3'] += 1

        total_count = len(test_df)
        # print(confusion_matrix)
        accuracy += (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[2, 2]) * 100 / total_count
    
        p0 = confusion_matrix.iloc[0, 0] * 100 / (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[2, 0])
        r0 = confusion_matrix.iloc[0, 0] * 100 / (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[0, 2])
        p1 = confusion_matrix.iloc[1, 1] * 100 / (confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[2, 1])
        r1 = confusion_matrix.iloc[1, 1] * 100 / (confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2])
        p2 = confusion_matrix.iloc[2, 2] * 100 / (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2] + confusion_matrix.iloc[2, 2])
        r2 = confusion_matrix.iloc[2, 2] * 100 / (confusion_matrix.iloc[2, 0] + confusion_matrix.iloc[2, 1] + confusion_matrix.iloc[2, 2])

        precision += (p0 + p1 + p2) / 3
        recall += (r0 + r1 + r2) / 3
        f1_score += 2 * ((p0 * r0) / (p0 + r0) + (p1 * r1) / (p1 + r1) + (p2 * r2) / (p2 + r2)) / 3

    accuracy /= k_fold
    precision /= k_fold
    recall /= k_fold
    f1_score /= k_fold
    y_accuracy[idx] = accuracy
    y_precision[idx] = precision
    y_recall[idx] = recall
    y_f1[idx] = f1_score
    idx += 1

print(y_accuracy)
print(y_precision)
print(y_recall)
print(y_f1)

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
