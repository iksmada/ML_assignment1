import operator
from decimal import Decimal
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


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


def gradient_descent(X, y, iterations=100, alpha=0.1, gamma=0.1, beta=None):
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
        gradient = (X.T.dot(loss) + gamma * beta) / m
        beta = beta - alpha * gradient
        cost = cost_function(X, y, beta)
        cost_history[iteration] = cost

        ## If you really want to merge everything in one line:
        # beta = beta - alpha * (X.T.dot(X.dot(beta)-y)/m)
        # cost = cost_function(X, y, beta)
        # cost_history[iteration] = cost

    return beta, cost_history


def cross_validation(X, y, iterations=50, alphas=(0.001, 0.01, 0.1, 1, 10), gammas=None, beta=None):
    if gammas is None:
        gammas = np.power(1 / 10, range(1, 15))

    best_cost = -1
    best_gamma = 0
    best_alpha = 0
    for alpha in alphas:
        for gamma in gammas:
            beta, cost_history = gradient_descent(X, y, iterations, alpha, gamma)
            cost = min(cost_history)
            if (best_cost == -1 or cost < best_cost):
                best_alpha = alpha
                best_gamma = gamma
                best_cost = cost
    return best_alpha, best_gamma


def feature_test(i, train_data_x, train_data_y, test_data_x, test_data_y):
    # add bias feature at the beginning
    feature = np.insert(train_data_x[:, i][:, None], [0], np.ones((train_data_x.shape[0], 1)), axis=1)
    testDataXOneFeat = np.insert(test_data_x[:, i][:, None], [0], np.ones((test_data_x.shape[0], 1)),
                                 axis=1)

    alpha, gamma = cross_validation(feature, train_data_y)

    # Train the model using the training sets
    beta, cost_history = gradient_descent(feature, train_data_y, 100, alpha, gamma)

    # Make predictions using the testing set
    predDataY = predict(testDataXOneFeat, beta)

    # The coefficients
    mse_pre = (mean_squared_error(test_data_y, predDataY))
    # The mean squared error
    print("Feature " + str(i))
    print("Best Alpha %f, Best Gamma %.2E" % (alpha, Decimal(gamma)))
    print("Mean squared error: %.2f"
          % mse_pre)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_data_y, predDataY))
    print("========================================================")

    return mse_pre


def feature_combination(size, mse_pre, train_data_x, train_data_y, test_data_x, test_data_y, iterations=100, return_cost=False):

    aux = mse_pre.copy()
    features = []
    for j in range(size):
        local_min = aux.index(min(aux))
        features.append(local_min)
        del aux[local_min]

    # add bias feature at the beginning
    train_data_x_selected_feat = np.insert(train_data_x[:, features], [0], np.ones((train_data_x.shape[0], 1)),
                                       axis=1)
    test_data_x_selected_feat = np.insert(test_data_x[:, features], [0], np.ones((test_data_x.shape[0], 1)),
                                      axis=1)

    alpha, gamma = cross_validation(train_data_x_selected_feat, train_data_y)

    # Train the model using the training sets
    beta, cost_history = gradient_descent(train_data_x_selected_feat, train_data_y, iterations, alpha, gamma)

    # Make predictions using the testing set
    pred_data_y = predict(test_data_x_selected_feat, beta)

    # The coefficients
    mse_pos = (mean_squared_error(test_data_y, pred_data_y))
    print("Using %d features" % size)
    print("Best Alpha %f, Best Gamma %.2E" % (alpha, Decimal(gamma)))
    # The mean squared error
    print("Mean squared error: %.2f"
          % mse_pos)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_data_y, pred_data_y))
    print("========================================================")
    if return_cost:
        return mse_pos, cost_history
    else:
        return mse_pos

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
testDataXScaled = scaler.transform(testDataX)

num_cores = multiprocessing.cpu_count()

try:
    with open("mse.txt", 'r') as f:
        mse_pre = [float(line) for line in f]
except FileNotFoundError:
    mse_pre = Parallel(n_jobs=num_cores)(
        delayed(feature_test)(i, trainDataXScaled, trainDataY, testDataXScaled, testDataY) for i in
        range(trainDataXScaled.shape[1]))

    with open("mse.txt", 'w') as f:
        for s in mse_pre:
            f.write(str(s) + '\n')
best_feature = mse_pre.index(min(mse_pre))
print("\nBest feature is %d" % best_feature)
print("========================================================")

mse_pos = [0, mse_pre[best_feature]]
mse_pos = mse_pos + Parallel(n_jobs=num_cores)(delayed(feature_combination)(size, mse_pre, trainDataXScaled, trainDataY, testDataXScaled, testDataY, 50) for size in range(2, trainDataXScaled.shape[1]))

best_n_features = mse_pos.index(min(mse_pos[1:]))
print("\nBest selection is with %d features" % best_n_features)
print("========================================================")

mse_pos, cost_history = feature_combination(best_n_features, mse_pre, trainDataXScaled, trainDataY, testDataXScaled, testDataY, 100, True)

plt.plot(range(len(cost_history)), cost_history, 'ro')
plt.show()
