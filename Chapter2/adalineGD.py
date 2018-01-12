import numpy as np


class AdalineGD:

    """
    ADAptive LInear NEuron Classifier.
    
    Parameters
    ------------

    eta: learning rate, between 0 and 1
    n_iter: number of passes over the training dataset
    random_state: int
    Random seed initialisator

    -----

    Attributes:

    w_: 1d-array. Weights after fitting
    cost_: list. Sum-of-squares cost function value in each epoch.

    """


    def __init__(self, eta=0.01, n_iter=50, random_state=1):

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        """Fit training data. 
        Parameters:

        X: {array-like} with shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y: {array-like}, with shape = [n_samples]
        Array of the target values.
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []
        # Pass over the values for n_iter times
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            continue
        return self

    def net_input(self, X):
        """Calculate net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X  # In this case, the activation function is an identity. In more complex cases, it is not.

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
