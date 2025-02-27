import numpy as np

class KNNRegression:

    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        dst = np.array([np.abs(self._distance(x, xtrain)) for xtrain in self.X])
        idx = np.argsort(dst)[:self.n_neighbors]
        y_pred = np.mean(self.y[idx])
        return y_pred

    def _distance(self, x, y):
        return np.linalg.norm(x-y)
    

if __name__ == '__main__':
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    y_train = np.array([2, 3, 4, 7, 8, 9])
    X_test  = np.array([[4, 5], [5, 6], [10, 11]])
    y_test  = np.array([5, 6, 7])

    # Initialize and fit the model
    knn = KNNRegression(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Test predictions
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)

