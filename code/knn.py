"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        for i in range(Xtest.shape[1]):
            dist1 = utils.euclidean_dist_squared(self.X, Xtest)
            sorted_distances = np.argsort(dist1, axis=0)
            yhat = self.y[sorted_distances[:min(self.k , len(self.X))]]

        mode = np.zeros(yhat.shape[1])
        # print(range(yhat.shape[1]))
        for i in range(yhat.shape[1]):
            mode[i] = utils.mode(yhat[:,i])
        # return yhat
        return mode

class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        self.X = Xcondensed
        self.y = ycondensed
