import numpy as np
import utils

class Kmedians:

    def __init__(self, k):
        self.k = k
        self.means = []

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = utils.euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        N, D = X.shape
        means = np.array(self.means)
        distances = utils.euclidean_dist_squared(X, means)
        objective_value = np.zeros(distances.shape[1])
        # objective_value = 0
        for j in range(distances.shape[1]):
            for i in range(distances.shape[0]):
                objective_value[j] += distances[i][j]
        min_object = sum(objective_value)
        return min_object
