# Alternative version of predict function
def predict(self, X):
    M, D = X.shape
    y = np.zeros(M)

    # GET VALUES FROM MODEL
    splitVariable = self.splitModel.splitVariable
    splitValue = self.splitModel.splitValue
    splitSat = self.splitModel.splitSat

    if splitVariable is None:
        # If no further splitting, return the majority label
        y = splitSat * np.ones(M)

    # the case with depth=1, just a single stump.
    elif self.subModel1 is None:
        return self.splitModel.predict(X)

    else:
        # Recurse on both sub-models
        j = splitVariable
        value = splitValue

        splitIndex1 = X[:,j] > value
        splitIndex0 = X[:,j] <= value
        
        # for m in range(M)
        # if (X[m,j] > value):
        #     splitIndex1 = X[m,j]
        # else:
        #     splitIndex0 = X[m,j]

        y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
        y[splitIndex0] = self.subModel0.predict(X[splitIndex0])

    return y
