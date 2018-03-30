import operator
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def oneFeatureAtATime(regr):
    mse = {}
    i = 0
    for feature in np.transpose(trainDataX):

        # Train the model using the training sets
        regr.fit(feature[:, None], trainDataY)

        # Make predictions using the testing set
        predDataY = regr.predict(testDataX[:, i][:, None])
        score = regr.score(testDataX[:, i][:, None], testDataY)

        print("feature " + str(i))
        # The coefficients
        # print('Coefficients: \n', regr.coef_)
        print('Coefficient of determination R^2: %.2f' % score)
        mse[i] = (mean_squared_error(testDataY, predDataY))
        # The mean squared error
        print("Mean squared error: %.2f"
              % mse[i])
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(testDataY, predDataY))
        print("========================================================")
        i = i + 1

    return mse


def featureCombination(regr, mse):
    for size in range(1, trainDataX.shape[1]):

        aux = mse.copy()
        features = []
        for i in range(size):
            localMax = max(aux.items(), key=operator.itemgetter(1))[0]
            features.append(localMax)
            del aux[localMax]

        # Train the model using the training sets
        regr.fit(trainDataX[:, features], trainDataY)

        # Make predictions using the testing set
        predDataY = regr.predict(testDataX[:, features])
        score = regr.score(testDataX[:, features], testDataY)

        print("Using %d feature " % size)
        # The coefficients
        # print('Coefficients: \n', regr.coef_)
        print('Coefficient of determination R^2: %.2f' % score)
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(testDataY, predDataY))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(testDataY, predDataY))
        print("========================================================")

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

# Create linear regression object
regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], normalize=True)

mse = oneFeatureAtATime(regr)

featureCombination(regr, mse)
