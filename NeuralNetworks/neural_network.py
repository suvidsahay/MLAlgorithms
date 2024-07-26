import numpy as np

class NeuralNetwork:
    def __init__(self, train_df, layers, neurons, r_lambda, alpha, y):
        self.layers = layers
        self.neurons = neurons
        self.y = y
        self.train_df = train_df
        self.r_lambda = r_lambda
        self.alpha = alpha
        self.n = len(self.train_df.iloc[0].to_numpy())
        # print(train_df.columns)
    
        self.A = [None] * (layers + 2)
        self.T = [None] * (layers + 1)
        self.delta = [None] * (layers + 2)
        self.D = [None] * (layers + 1)
        if self.layers == 0:
            self.T[0] = np.random.uniform(-1, 1, size=(len(y[0]), self.n + 1))
            self.D[0] = np.zeros((len(y[0]), self.n + 1))
        else :
            self.T[0] = np.random.uniform(-1, 1, size=(self.neurons[0], self.n + 1))
            self.D[0] = np.zeros((self.neurons[0], self.n + 1))
            
            for i in range(layers - 1):
                self.T[i + 1] = np.random.uniform(-1, 1, size=(self.neurons[i + 1], self.neurons[i] + 1))
                self.D[i + 1] = np.zeros((self.neurons[i + 1], self.neurons[i] + 1))
            self.T[layers] = np.random.uniform(-1, 1, size=(len(y[0]), self.neurons[layers - 1] + 1))
            self.D[layers] = np.zeros((len(y[0]), self.neurons[layers - 1] + 1))

    def fit(self):
        e = 0.01
        J = 1e7
        J_new = 0
        itr = 0
        mod = 10
        while abs(J - J_new) > e:
            J = J_new
            J_new = 0
            self.D = [None] * (self.layers + 1)
            self.A = [None] * (self.layers + 2)

            if self.layers == 0:
                self.D[0] = np.zeros((len(self.y[0]), self.n + 1))
            else :
                self.D[0] = np.zeros((self.neurons[0], self.n + 1))
                for l in range(self.layers - 1):
                    self.D[l + 1] = np.zeros((self.neurons[l + 1], self.neurons[l] + 1))
                self.D[self.layers] = np.zeros((len(self.y[0]), self.neurons[self.layers - 1] + 1))

            for i in range(len(self.train_df)):
                #forward propagation
                x = self.train_df.iloc[i].to_numpy()
                self.A[0] = np.append([1], x)
                for j in range(self.layers):
                    self.A[j + 1] = np.append([1], self.g(np.matmul(self.T[j], self.A[j])))
                self.A[self.layers + 1] = self.g(np.matmul(self.T[self.layers], self.A[self.layers]))
                J_new += self.cost(self.A[self.layers + 1], self.y[i])
                
                #bakward propagation
                self.delta[self.layers + 1] = self.A[self.layers + 1] - self.y[i]
                for j in range(self.layers, 0, -1):
                    # print(i, j, T[j].shape, delta[j + 1].shape, A[j].shape)
                    self.delta[j] = np.multiply(np.matmul(np.transpose(self.T[j]), self.delta[j + 1]), np.multiply(self.A[j], 1 - self.A[j]))
                    self.delta[j] = np.delete(self.delta[j], 0)
                    # print(delta[j].shape, np.transpose(delta[j + 1][np.newaxis]).shape, A[j][np.newaxis].shape)
                    self.D[j] += np.matmul(np.transpose(self.delta[j + 1][np.newaxis]), self.A[j][np.newaxis])
                self.D[0] += np.matmul(np.transpose(self.delta[1][np.newaxis]), self.A[0][np.newaxis])
            
            for j in range(self.layers, -1, -1):
                self.D[j] = (self.D[j] + np.multiply(self.r_lambda, self.T[j])) / len(self.train_df)
                self.T[j] -= np.multiply(self.alpha, self.D[j])
                # print(itr, j, self.T[j])
            
            J_new += self.regularisation(self.T, self.r_lambda)
            J_new /= len(self.train_df)
            # if itr % (mod / 10) == 0:
            #     print("Iteration", itr, J, J_new)
            # if itr == mod:
            #     mod *= 10;
            # itr += 1
    
    def test(self, test_df):
        #forward propagation
        x = test_df.to_numpy()
        self.A[0] = np.append([1], x)
        for j in range(self.layers):
            # print(j, self.A[j].shape, self.T[j].shape)
            self.A[j + 1] = np.append([1], self.g(np.matmul(self.T[j], self.A[j])))
        self.A[self.layers + 1] = self.g(np.matmul(self.T[self.layers], self.A[self.layers]))
        # print(self.A[self.layers + 1])
        return np.argmax(self.A[self.layers + 1])
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, matrix):
        res = [None] * (len(matrix))
        for i in range(len(matrix)):
            res[i] = self.sigmoid(matrix[i])
        return res
                
    def cost(self, A, y):
        return -1 * np.sum(np.multiply(y, np.log(A)) + np.multiply((1 - y), np.log(np.maximum(1e-10, 1 + np.multiply(-1, A)))))

    def regularisation(self, T, r_lambda):
        return (sum(np.sum(np.square(matrix[:, 1:])) for matrix in T) * r_lambda) / 2



