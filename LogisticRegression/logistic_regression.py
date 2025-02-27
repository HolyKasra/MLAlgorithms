
import numpy as np

class LogisticRegression:

    def __init__(self, 
                lr=0.1, 
                max_iter=1000, 
                thresh=0.5, 
                random_state=2025,
                fit_intercept=True):
        
        self.lr = lr
        self.max_iter = max_iter
        self.thresh = thresh
        self.rng = np.random.RandomState(seed=random_state)
        self.fit_intercept = fit_intercept
        self.loss = []

    def fit(self, X_train, y_train):

        if self.fit_intercept:
            X_train = np.c_[np.ones(X_train.shape[0]), X_train]

        n_features = X_train.shape[1]
        self.w = self.rng.randn(n_features)

        # Training loop
        for _ in range(self.max_iter):

            z = X_train.dot(self.w)
            y_pred = self._sigmoid(z)

            # Calculating Gradient
            gradient = X_train.T.dot(y_pred - y_train)

            # Updating weights
            self.w -= self.lr * gradient

            # Loss calculation
            loss_val = self._loss(y_train, y_pred)
            self.loss.append(loss_val)

    def predict(self, X_test):
        if self.fit_intercept:
            X_test = np.c_[np.ones(X_test.shape[0]), X_test]

        y_pred = X_test.dot(self.w)
        y_prob = self._sigmoid(y_pred)
        y_pred_class = np.where(y_prob > self.thresh, 1, 0)

        return y_pred_class

    def _sigmoid(self, x):
        x = np.clip(x, -100,100)
        return 1 / (1 + np.exp(-x))

    def _loss(self, y_true, y_pred, eps=1e-16):
        return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)).mean()
    

if __name__ == '__main__':
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y= True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(f'My Model Accuracy: {accuracy_score(y_test, y_pred)}')