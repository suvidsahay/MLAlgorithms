from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random_forest
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def divide (a, b):
    if b == 0:
        return 0
    else:
        return a / b

def shuffle(df):
    df_copy = df.copy()
    df_copy = df_copy.sample(frac = 1).reset_index(drop=True)

    return df_copy

def k_fold_partition(df, k, i, output):
    # print(df[output])    
    df_0 = df[df[output_class] == 0]
    df_1 = df[df[output_class] == 1]

    # print(len(df_0), len(df_1))

    # print(df['class'].value_counts())    

    test_df = pd.concat([df_0[int((len(df_0) * i) / k) : int((len(df_0) * (i + 1)) / k)], 
                         df_1[int((len(df_1) * i) / k) : int((len(df_1) * (i + 1)) / k)]])
    train_df = df.drop(test_df.index)

    return train_df, test_df


df = pd.read_csv('datasets/titanic.csv')
df = df.drop(columns=['Name'])
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})


attr = [['Pclass', False],
        ['Sex', False],
        ['Age', True],
        ['Siblings/Spouses Aboard', True],
        ['Parents/Children Aboard', True],
        ['Fare', True]]
output_class = 'Survived'

shuffle(df)

k_fold = 10
ntree=[1, 5, 10, 20, 30, 40, 50]

iterations = len(ntree)
y_accuracy = np.zeros(iterations)
y_precision = np.zeros(iterations)
y_recall = np.zeros(iterations)
y_f1 = np.zeros(iterations)
idx=0

for iter in ntree:
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    total_confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])

    for k in range(k_fold):
        train_df, test_df = k_fold_partition(df, k_fold, k, output_class)
        # print(train_df)
        # print(test_df)

        rf = random_forest.generate_random_forest(train_df, attr, iter, output_class)
        confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])


        for i in range(len(test_df)):
            output = random_forest.majority_class(rf, test_df.iloc[i])
            # print(output, test_df[output_class].iloc[i])
            if output == test_df[output_class].iloc[i]:
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
    y_accuracy[idx] = accuracy
    y_precision[idx] = precision
    y_recall[idx] = recall
    y_f1[idx] = f1_score
    idx += 1

    print(f"For Random Forest with {iter} trees:", accuracy, precision, recall, f1_score)
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

