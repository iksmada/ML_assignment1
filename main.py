from decimal import Decimal
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    # number of training examples
    m = len(y)

    # Calculate the cost with the given parameters
    J = (np.sum((predict(X, beta) - y) ** 2) + beta[:, 0].dot(beta)) / 2 / m

    return float(J)


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

def normal_equation(X, y):

    x_transpose = X.T
    beta = np.linalg.pinv(x_transpose.dot(X)).dot(x_transpose).dot(y)
    return beta


def cross_validation(X, y, iterations=50, alphas=(0.001, 0.01, 0.1, 1, 10), gammas=None, beta=None):
    if gammas is None:
        gammas = np.concatenate((np.power(10, range(5, 0, -1)), np.power(1 / 10, range(0, 15))))

    best_cost = -1
    best_gamma = 0
    best_alpha = 0
    for alpha in alphas:
        for gamma in gammas:
            beta = normal_equation(X, y)
            cost = float(min(cost_history))
            if best_cost == -1 or cost < best_cost:
                best_alpha = alpha
                best_gamma = gamma
                best_cost = cost
    return best_alpha, best_gamma


def feature_test(i, train_data_x, train_data_y, test_data_x, test_data_y):
    # add bias feature at the beginning
    feature = np.insert(train_data_x[:, i][:, None], [0], np.ones((train_data_x.shape[0], 1)), axis=1)
    test_data_x_one_feat = np.insert(test_data_x[:, i][:, None], [0], np.ones((test_data_x.shape[0], 1)), axis=1)

    # Train the model using the training sets
    beta = normal_equation(feature, train_data_y)
    cost = cost_function(feature, train_data_y, beta)
    # Make predictions using the testing set
    pred_data_y = predict(test_data_x_one_feat, beta)

    # The coefficients
    mse_pre = (mean_squared_error(test_data_y, pred_data_y))
    # The mean squared error
    print("Feature " + str(i))

    show_stats(cost, pred_data_y, test_data_y)

    return cost


def feature_combination(size, mse_pre, train_data_x, train_data_y, test_data_x, test_data_y):
    aux = list(mse_pre)
    features = []
    for j in range(size):
        local_min = aux.index(min(aux))
        features.append(local_min + j)
        aux.remove(min(aux))

    # add bias feature at the beginning
    train_data_x_selected_feat = np.insert(train_data_x[:, features], [0], np.ones((train_data_x.shape[0], 1)),
                                           axis=1)
    test_data_x_selected_feat = np.insert(test_data_x[:, features], [0], np.ones((test_data_x.shape[0], 1)),
                                          axis=1)

    # Train the model using the training sets
    beta = normal_equation(train_data_x_selected_feat, train_data_y)
    cost = cost_function(train_data_x_selected_feat, train_data_y, beta)
    # Make predictions using the testing set
    pred_data_y = predict(test_data_x_selected_feat, beta)

    print("Using features:" +str(features))
    mae_pos = show_stats(cost, pred_data_y, test_data_y)

    return mae_pos


def show_stats(cost_min, pred_data_y, test_data_y):
    # The coefficients
    mse_pos = (mean_squared_error(test_data_y, pred_data_y))
    mae_pos = (mean_absolute_error(test_data_y, pred_data_y))
    # The mean squared error
    print("Best Cost J(theta): %f" % cost_min)
    print("Mean Absolute error: %.2f" % mae_pos)
    print("Mean squared error: %.2f"
          % mse_pos)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(test_data_y, pred_data_y))
    print("========================================================")
    return mae_pos


def remove_noise(train_data_x, train_data_y):
    y, x, _ = plt.hist(train_data_y.ravel(), range(0, 10000, 100))
    share_center = x[np.where(y == y.max())[0]]
    share_std = np.std(train_data_y)
    sample_del = []
    for i in range(size_before):
        if train_data_y[i][0] > 0.2 * share_std + share_center or train_data_y[i][0] < share_center - 0.05 * share_std:
            sample_del.append(i)
    train_data_y = np.delete(train_data_y, sample_del, axis=0)
    train_data_x = np.delete(train_data_x, sample_del, axis=0)

    return train_data_x, train_data_y


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
testDataY = np.loadtxt(rawData, skiprows=1)[:, None]

# do Standart Scaling
scaler = StandardScaler()
trainDataX = scaler.fit_transform(trainDataX)
testDataX = scaler.transform(testDataX)

size_before = trainDataY.shape[0]

trainDataX, trainDataY = remove_noise(trainDataX, trainDataY)

print("Deleted %d samples" % (size_before - trainDataY.shape[0]))
plt.hist(trainDataY.ravel(), range(0, 10000, 100))
plt.show(block=True)

num_cores = multiprocessing.cpu_count()

try:
    with open("cost2.txt", 'r') as f:
        cost_pre = [float(line) for line in f]
except FileNotFoundError:
    cost_pre = Parallel(n_jobs=num_cores)(
        delayed(feature_test)(i, trainDataX, trainDataY, testDataX, testDataY) for i in
        range(trainDataX.shape[1]))

    with open("cost2.txt", 'w') as f:
        for s in cost_pre:
            f.write(str(s) + '\n')
best_feature = cost_pre.index(min(cost_pre))
print("\nBest feature is %d" % best_feature)
print("========================================================")

mae_pos = [0]
mae_pos = mae_pos + Parallel(n_jobs=num_cores)(
    delayed(feature_combination)(size, cost_pre, trainDataX, trainDataY, testDataX, testDataY) for size
    in range(1, trainDataX.shape[1]))

best_n_features = mae_pos.index(min(mae_pos[1:]))
print("\nBest selection is with %d features" % best_n_features)
print("========================================================")

cost_final = feature_combination(best_n_features, cost_pre, trainDataX, trainDataY, testDataX, testDataY)
mae_pos[0] = cost_final

plt.plot(range(len(mae_pos)), mae_pos, 'ro')
plt.show()
