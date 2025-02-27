import numpy as np

class SVClassifier:
    '''
    This is a hard margin implementation of support vector machines algorithm!
    '''
    def __init__(self, l2=1, lr=0.01, max_iter=1000, random_state=42):

        self.l2 = l2
        self.lr = lr
        self.max_iter = max_iter
        self.w  = None
        self.b  = None
        self.rng = np.random.RandomState(random_state)


    def fit(self, X, y):
        # Initialization!
        n_sample, n_feature = X.shape
        self.w = self.rng.normal(size=n_feature)
        self.b = self.rng.normal()

        # Vectorized Loop
        for _ in range(self.max_iter):
            condition = y*(X.dot(self.w) + self.b)
            grad = self.l2 * self.w - 1/n_sample * X[condition<1].T.dot(y[condition<1])
            self.w -= self.lr * grad
            self.b += self.lr/n_sample *(y[condition<1].sum())
  
    def predict(self, X):
        decision = X.dot(self.w) + self.b
        return np.sign(decision)

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_blobs(n_samples=2000, 
                    n_features=2, 
                    centers=2,
                    cluster_std=1, 
                    random_state=42)

    y = np.where(y==0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    sv = SVClassifier()
    sv.fit(X_train, y_train)

    y_pred = sv.predict(X_test)
    print(f'My SVC Accuracy: {accuracy_score(y_test, y_pred)*100}%')