from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import neural_networks
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

for layers in range(1, 4, 1):
    for neurons in (2 ** x for x in range(4, 6, 1)):
        accuracy = 0
        precision = 0
        recall = 0
        f1score = 0
        total_confusion_matrix = pd.DataFrame(np.zeros((10, 10)), index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'])

        for k in range(k_fold):
            a = 0
            p = [None] * 10
            r = [None] * 10
            f1 = [None] * 10
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

            nn = neural_networks.NeuralNetwork(train_df, layers, [neurons] * layers, 0.25, 0.5, y_train)
            nn.fit()

            correct_count = 0
            confusion_matrix = pd.DataFrame(np.zeros((10, 10)), index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], columns=['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'])
            for i in range(len(test_df)):
                output = nn.test(test_df.iloc[i])
                expected = y_test.iloc[i]
                confusion_matrix.iloc[expected, output] += 1
            # print(confusion_matrix)
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
            f1score += np.average(f1)

            # print(confusion_matrix)
            total_confusion_matrix += confusion_matrix
        accuracy /= k_fold
        precision /= k_fold
        recall /= k_fold
        f1score /= k_fold


        print(f"For Neural network with {layers} hidden layers and {neurons} neurons in each layer:", accuracy * 100, precision, recall, f1score)
        print(total_confusion_matrix)




# digit_to_show = np.random.choice(range(N), 1)[0]
# print("Attributes:", digits_dataset_X[digit_to_show])
# print("Class:", digits_dataset_y[digit_to_show])
# plt.imshow(np.reshape(digits_dataset_X[digit_to_show],(8,8)))
# plt.show()
