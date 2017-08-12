import numpy as np
import utils

from decision_stump import DecisionStump
from random_tree import RandomTree

class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.models = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.models = []
        for i in range(self.num_trees):
            self.models = np.append(self.models, RandomTree(max_depth=self.max_depth))
            self.models[i].fit(X, y)

    def predict(self, X):
        y_pred = []
        for i in range(self.num_trees):
            y_pred.append(self.models[i].predict(X))

        result = np.array(y_pred)
        mode = np.zeros(result.shape[1])
        for j in range(result.shape[1]):
            mode[j] = utils.mode(result[:,j])
        return mode
