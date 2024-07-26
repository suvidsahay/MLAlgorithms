from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import neural_networks
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def normalise(train_df, test_df):
    numerical = ['remainder__Age', 'remainder__Siblings/Spouses Aboard', 'remainder__Parents/Children Aboard', 'remainder__Fare']
    max_values = train_df[numerical].max()
    min_values = train_df[numerical].min()

    ranges = max_values - min_values
    ranges[ranges == 0] = 1

    train_df[numerical] = 2 * (train_df[numerical] - min_values) / ranges - 1
    test_df[numerical] = 2 * (test_df[numerical] - min_values) / ranges - 1

    return train_df, test_df


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


df = pd.read_csv('datasets/titanic.csv')
df = df.drop(columns=['Name'])


attr = ['Pclass', 'Sex']

transformer = make_column_transformer(
    (OneHotEncoder(), attr),
    remainder='passthrough')

transformed = transformer.fit_transform(df)

df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
output_class = 'remainder__Survived'

shuffle(df)

k_fold = 10

for layers in range(1, 5, 1):
    for neurons in (2 ** x for x in range(1, 6, 1)):
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        total_confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])

        for k in range(k_fold):
            train_df, test_df = k_fold_partition(df, k_fold, k, output_class)
            # print(len(train_df))
            # print(len(test_df))
            y_train = train_df[output_class]
            # print(np.unique(y_train, return_counts=True))
            y_train = np.zeros((len(train_df), y_train.nunique()))

            for i in range(len(train_df)):
                # print(train_df[output_class].iloc[i])
                y_train[i][int(train_df[output_class].iloc[i])] = 1
            
            train_df = train_df.drop(columns=[output_class])


            # print(train_df.head())
            y_test = test_df[output_class]
            test_df = test_df.drop(columns=output_class)

            # print(train_df.shape)
            # print(test_df.shape)

            train_df, test_df = normalise(train_df, test_df)


            nn = neural_networks.NeuralNetwork(train_df, layers, [neurons] * layers, 0.01, 1, y_train)
            nn.fit()

            correct_count = 0
            confusion_matrix = pd.DataFrame(np.zeros((2, 2)), index=[0, 1], columns=['pred_pos', 'pred_neg'])


            for i in range(len(test_df)):
                output = nn.test(test_df.iloc[i])
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


# digit_to_show = np.random.choice(range(N), 1)[0]
# print("Attributes:", digits_dataset_X[digit_to_show])
# print("Class:", digits_dataset_y[digit_to_show])
# plt.imshow(np.reshape(digits_dataset_X[digit_to_show],(8,8)))
# plt.show()
