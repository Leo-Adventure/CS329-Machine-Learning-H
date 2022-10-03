from audioop import rms
import numpy as np
import itertools
import functools


class PolynomialFeatures:
    """
    polynomial features
    transforms input array with polynomial features
    Example
    =======
    x =
    [[a, b],
    [c, d]]
    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features
        Parameters
        ----------
        degree : int
        degree of polynomial
        """
        self.degree = degree

    def fit_transform(self, X):
        """
        transforms input array with polynomial features
        Parameters
        ----------
        X : (sample_size, n) ndarray
        input array
        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
        polynomial features
        """
        features = [np.ones(len(X))]
        X = X[np.newaxis, :]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(X, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


class LinearRegression:
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def __init__(self):
        self.w = None
        self.var = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        perform least squares fitting
        Parameters
        ----------
        X : (N, D) np.ndarray
        training independent variable
        t : (N,) np.ndarray
        training dependent variable
        """
        # write your code here
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))


    def predict(self, X: np.ndarray):
        """
        make prediction given input
        Parameters
        ----------
        X : (N, D) np.ndarray
        samples to predict their output
        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        """
        # write your code here
        y = X @ self.w
        return y

def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


def main():
    n, m = input().split()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        x, y = input().split()
        x_train.append(float(x))
        y_train.append(float(y))
    for _ in range(int(m)):
        x, y = input().split()
        x_test.append(float(x))
        y_test.append(float(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # write your code here

    results = [] 
    
    for K in range(11):
        pn = PolynomialFeatures(K)
        X_TRAIN = pn.fit_transform(x_train)
        X_TEST = pn.fit_transform(x_test) 
        model = LinearRegression()
        model.fit(X_TRAIN, y_train)
        predict = model.predict(X_TEST)
        value = rmse(predict, y_test)
        results.append((value, model, K))

    p = min(results)
    print (p[2])
    print (((p[1].var)).round(6)) 


if __name__ == '__main__':
    main()
