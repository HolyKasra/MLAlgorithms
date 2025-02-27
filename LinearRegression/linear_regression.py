import numpy as np

# It also handles ridge and lasso regression by tuning the l2-l1 paremeters!
class LinearRegressor:

    def __init__(self, 
                lr=0.01, 
                epochs=10000, 
                lasso_l1=0, 
                ridge_l2=0,  
                seed=42, 
                fit_intercept=True):
        
        # Hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept

        # Regularization parameters
        self.lasso_l1 = lasso_l1
        self.ridge_l2 = ridge_l2
        
        # Loss Cache
        self.loss = []

        # For replication
        self.rng = np.random.RandomState(seed=seed)

    def fit(self, X, y):

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0],), X]

        n_sample, n_feature = X.shape
        self.w = self.rng.randn(n_feature)  # Random weight initialization

        for i in range(self.epochs):
            ypred = X.dot(self.w)
            self.w -= self.lr / n_sample * (X.T.dot(ypred - y) + 
            self.ridge_l2*self.w +
            self.lasso_l1*np.sign(self.w))  
            
            # Optionally compute and store the loss for each epoch
            loss = self._loss_fcn(y, ypred)
            self.loss.append(loss)
            
    def predict(self, X_test):
        if self.fit_intercept:
            X_test = np.c_[np.ones(X_test.shape[0],), X_test]
        return X_test.dot(self.w)
    
    @staticmethod
    def _loss_fcn(y, ypred):
        return ((y-ypred)**2).mean()
    

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    X, y = make_regression(n_samples=200, 
                       n_features=10, 
                       n_informative=2,
                       noise=1, 
                       random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.5, 
                                                        random_state=42)
    
    reg = LinearRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print(f'My Model Rsquared: {r2_score(y_test, y_pred)}')