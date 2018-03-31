from decimal import Decimal

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def predict(X, beta):
    return X.dot(beta)

def cost_function(X, y, beta):
    """
    cost_function(X, y, beta) computes the cost of using beta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y)

    ## Calculate the cost with the given parameters
    J = (np.sum((predict(X, beta) - y) ** 2) + beta[:, 0].dot(beta)) / 2 / m

    return J


def gradient_descent(X, y, iterations=1000, alpha=0.1, gamma = 0.1, beta=None):
    """
    gradient_descent() performs gradient descent to learn beta by
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    m = len(y)
    if beta is None:
        beta = np.zeros((X.shape[1], 1))

    for iteration in range(iterations):
        hypothesis = predict(X, beta)
        loss = hypothesis - y
        gradient = (X.T.dot(loss) + gamma * beta)/ m
        beta = beta - alpha * gradient
        cost = cost_function(X, y, beta)
        cost_history[iteration] = cost

        ## If you really want to merge everything in one line:
        # beta = beta - alpha * (X.T.dot(X.dot(beta)-y)/m)
        # cost = cost_function(X, y, beta)
        # cost_history[iteration] = cost

    return beta, cost_history


def cross_validation(X, y, iterations=10, alphas=(0.001, 0.01, 0.1, 1, 10), gammas=None, beta=None):

    if gammas is None:
        gammas = np.power(1/10, range(1, 10))
    if iterations is None:
        iterations = 100

    best_cost = -1
    best_gamma = 0
    best_alpha = 0
    for alpha in alphas:
        for gamma in gammas:
            beta, cost_history = gradient_descent(X, y, iterations, alpha, gamma)
            cost = min(cost_history)
            if (best_cost == -1 or cost<best_cost):
                best_alpha = alpha
                best_gamma = gamma
                best_cost = cost
    return best_alpha, best_gamma


# Load the shares dataset
rawData = open("dataset/shares/train.csv")
trainData = np.genfromtxt(rawData, skip_header=1, delimiter=',')

trainDataX = trainData[:, 2:-1]  # select columns 2 until end, take off unpredictable features, and shares
trainDataY = trainData[:, -1:]  # select last column, the number of shares

# Load the shares testset
rawData = open("dataset/shares/test.csv")
testData = np.genfromtxt(rawData, skip_header=1, delimiter=',')
testDataX = testData[:, 2:]  # take off unpredictable features

# Load the shares testset expected result
rawData = open("dataset/shares/test_target.csv")
testDataY = np.loadtxt(rawData, skiprows=1)

# do Standart Scaling
scaler = StandardScaler()
trainDataXScaled = scaler.fit_transform(trainDataX)
# add bias feature at the beginning
trainDataXScaled = np.insert(trainDataXScaled, [0], np.ones((trainDataXScaled.shape[0], 1)), axis=1)
testDataXScaled = scaler.transform(testDataX)
testDataXScaled = np.insert(testDataXScaled, [0], np.ones((testDataXScaled.shape[0], 1)), axis=1)


alpha, gamma = cross_validation(trainDataXScaled, trainDataY)

print("Best Alpha %f, Best Gamma %.2E" % (alpha, Decimal(gamma)))

beta, cost_history = gradient_descent(trainDataXScaled, trainDataY, 1000, alpha, gamma)

predDataY = predict(testDataXScaled, beta)

print("Mean squared error: %.2f"% mean_squared_error(testDataY, predDataY))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testDataY, predDataY))
print("========================================================")

plt.plot(range(len(cost_history)), cost_history, 'ro')
plt.show()

