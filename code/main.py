import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import math

import utils

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from matplotlib.font_manager import FontProperties
from random_forest import RandomForest

from knn import KNN, CNN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":

        # 1. Load fluTrends dataset
        data = utils.load_dataset("fluTrends")
        X = data[0]
        y = data[1]

        # minimum , max, mean, median, mode,
        mode = utils.mode(X)
        mean = np.mean(X)
        median = np.median(X)
        minimum = np.amin(X)
        maximum = np.amax(X)
        print("Mode: %.5f" % mode)
        print("Mean: %.5f" % mean)
        print("Median: %.5f" % median)
        print("Minimum: %.5f" % minimum)
        print("Maximum: %.5f" % maximum)
        #pass

    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2.1_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)

        print(y_pred)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "2.4":
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, my_tree_errors, label="mine")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))


        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q24treeerrors.pdf")
        plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        depths = np.arange(1, 15)
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        # Training Error Plot
        t = time.time()
        training_errors = np.zeros(depths.size)
        test_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)

            model.fit(X, y)

            y_pred = model.predict(X)
            y_pred_test = model.predict(X_test)

            training_errors[i] = np.mean(y_pred != y)
            test_errors[i] = np.mean(y_pred_test != y_test)

        plt.plot(depths, training_errors, label="Training error")
        plt.plot(depths, test_errors, label="Test error")

        fontP = FontProperties()
        fontP.set_size('small')
        plt.xlabel("Depth of tree")
        plt.ylabel("Error")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, prop = fontP)
        fname = os.path.join("..", "figs", "q31errorplot.pdf")
        plt.savefig(fname)

    elif question == "3.2":
        depths = np.arange(1, 15)
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        # model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        # model.fit(X, y)
        training_set_X = X[0:len(X)/2]
        training_set_y = y[0:len(y)/2]
        validation_set_X = X[len(X)/2:]
        validation_set_y = y[len(y)/2:]

        # Switched Training Error Plot (2nd n/2 examples as training set)
        t = time.time()
        training_set_errors = np.zeros(depths.size)
        validation_set_errors = np.zeros(depths.size)

        switched_training_set_errors = np.zeros(depths.size)
        switched_validation_set_errors = np.zeros(depths.size)

        for i, max_depth in enumerate(depths):
            # Create first regular model and switched model
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            switched_model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)

            # Fit these models separately with the different sets
            model.fit(validation_set_X, validation_set_y)
            switched_model.fit(training_set_X, training_set_y)

            # Predicted y for the first and second sets
            y_pred = model.predict(validation_set_X)
            switched_y_pred = switched_model.predict(training_set_X)

            # Determine the errors for both cases
            training_set_errors[i] = np.mean(y_pred != training_set_y)
            validation_set_errors[i] = np.mean(y_pred != validation_set_y)
            switched_training_set_errors[i] = np.mean(switched_y_pred != training_set_y)
            switched_validation_set_errors[i] = np.mean(switched_y_pred != validation_set_y)

        # Plot each case
        plt.plot(depths, training_set_errors, label="Training Set error")
        plt.plot(depths, validation_set_errors, label="Validation Set error")
        plt.plot(depths, switched_training_set_errors, label="Switched Training Set error")
        plt.plot(depths, switched_validation_set_errors, label="Switched Validation Set error")

        fontP = FontProperties()
        fontP.set_size('small')
        plt.xlabel("Depth of tree")
        plt.ylabel("Error")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, prop = fontP)
        fname = os.path.join("..", "figs", "q32errorplot.pdf")
        plt.savefig(fname)


    if question == '4.1':
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = KNN(k=5)
        model.fit(X, y)

        y_pred_tr = model.predict(X)
        tr_error = np.mean(y_pred_tr != y)

        y_pred_te = model.predict(X_test)
        te_error = np.mean(y_pred_te != y_test)

        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

        utils.plotClassifier(model, X, y)

    if question == '4.2':
        dataset = utils.load_dataset('citiesBig1')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        model = CNN(1)
        model.fit(X,y)

        t = time.time()
        y_pred = model.predict(X)
        print("Time to predict: %f" % (time.time()-t))
        te_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        tr_error = np.mean(y_pred != y_test)

        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)
        print(y_pred.shape[0])

        utils.plotClassifier(model, X, y)

    if question == '5':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("n = %d, d = %d" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            t = time.time()
            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)
            print("Time taken: %f" % (time.time()-t))


        evaluate_model(DecisionTree(max_depth=50, stump_class=DecisionStumpInfoGain))
        evaluate_model(RandomTree(max_depth=50))
        evaluate_model(RandomForestClassifier(n_estimators=50, max_depth=50, criterion='entropy', random_state=1))
        evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))
