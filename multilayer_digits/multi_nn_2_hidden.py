import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('multilayer_digits/datasets/mnist_train.csv')
test_data = pd.read_csv('multilayer_digits/datasets/mnist_test.csv')


data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
test_data = np.array(test_data)

test_data = test_data.T
labels_test = test_data[0]
test_samples = test_data[1:]
test_samples = test_samples / 255.

train_data = data[:m].T
labels_train = train_data[0]
train_samples = train_data[1:n]
train_samples = train_samples / 255.
_,m_train = train_samples.shape



class MultilayerMNIST():
    def __init__(self, sizes, epochs=200, rate=0.1):
        self.input_size = sizes[0]
        self.layer_1_size = sizes[1]
        self.layer_2_size = sizes[2]
        self.output_size = sizes[3]
        self.epochs = epochs
        self.rate = rate
        self.weights = self.set_weights()
    
    def set_weights(self):
        w1 = np.random.rand(self.layer_1_size, self.input_size) - 0.5
        b1 = np.random.rand(self.layer_1_size, 1) - 0.5
        w2 = np.random.rand(self.layer_2_size, self.layer_1_size) - 0.5
        b2 = np.random.rand(self.layer_2_size, 1) - 0.5
        w3 = np.random.rand(self.output_size, self.layer_2_size) - 0.5
        b3 = np.random.rand(self.output_size, 1) - 0.5
        return w1, b1, w2, b2, w3, b3


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def softmax_derivative(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0

    def forward(self, w1, b1, w2, b2, w3, b3, X):
        input_weighted = w1.dot(X) + b1
        activated_1 = self.ReLU(input_weighted)

        hidden_weighted = w2.dot(activated_1) + b2
        activated_2 = self.ReLU(hidden_weighted)

        Z2 = w3.dot(activated_2) + b3
        out_activated = self.softmax(Z2)

        return input_weighted, activated_1, hidden_weighted, activated_2, Z2, out_activated

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward(self, input_weighted, activated_1, hidden_weighted, activated_2, Z2, out_activated, w1, w2, w3, X, Y):
        one_hot_Y = self.one_hot(Y)

        dZ2 = out_activated - one_hot_Y
        delta_w3 = 1 / m * dZ2.dot(activated_2.T)
        delta_bias3 = 1 / m * np.sum(dZ2)

        delta_hidden_weighted = w3.T.dot(dZ2) * self.ReLU_deriv(hidden_weighted)
        delta_w2 = 1 / m * delta_hidden_weighted.dot(activated_1.T)
        delta_bias2 = 1 / m * np.sum(delta_hidden_weighted)

        delta_input_weighted = w2.T.dot(delta_hidden_weighted) * self.ReLU_deriv(input_weighted)
        delta_w1 = 1 / m * delta_input_weighted.dot(X.T)
        delta_bias1 = 1 / m * np.sum(delta_input_weighted)
        return delta_w1, delta_bias1, delta_w2, delta_bias2, delta_w3, delta_bias3

    def update_weights(self, w1, b1, w2, b2, w3, b3, delta_w1, delta_bias1, delta_w2, delta_bias2, delta_w3, delta_bias3, alpha):
        w1 -= alpha * delta_w1
        b1 -= alpha * delta_bias1    
        w2 -= alpha * delta_w2  
        b2 -= alpha * delta_bias2    
        w3 -= alpha * delta_w3 
        b3 -= alpha * delta_bias3   
        return w1, b1, w2, b2, w3, b3

    def get_predictions(self,output):
        return np.argmax(output, 0)

    def get_accuracy(self,predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def make_predictions(self, X, w1, b1, w2, b2, w3, b3):
        _, _, _, _, _, out_activated = self.forward(w1, b1, w2, b2, w3, b3, X)
        predictions = self.get_predictions(out_activated)
        return predictions


    def train(self, X, Y):
        w1, b1, w2, b2, w3, b3 = self.weights
        for i in range(self.epochs):
            input_weighted, activated_1, hidden_weighted, activated_2, Z2, out_activated = self.forward(w1, b1, w2, b2, w3, b3, X)
            delta_w1, delta_bias1, delta_w2, delta_bias2, delta_w3, delta_bias3 = self.backward(input_weighted, activated_1, hidden_weighted, activated_2, Z2, out_activated, w1, w2, w3, X, Y)
            w1, b1, w2, b2, w3, b3 = self.update_weights(w1, b1, w2, b2, w3, b3, delta_w1, delta_bias1, delta_w2, delta_bias2, delta_w3, delta_bias3, self.rate)
            if i % 50 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(out_activated)
                print(self.get_accuracy(predictions, Y))
        return w1, b1, w2, b2, w3, b3





epochs_n = 500
print(f'TEST {1}: sample size - {50000}, epochs - {epochs_n}')
network = MultilayerMNIST(sizes=[784, 60, 20, 10], epochs=500)
w1, b1, w2, b2, w3, b3 = network.train(X=train_samples, Y=labels_train)


test_predictions = network.make_predictions(test_samples, w1, b1, w2, b2, w3, b3)
print('Test sample accuracy:', network.get_accuracy(test_predictions, labels_test))



    