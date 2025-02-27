import numpy as np
from sklearn import tree

class GradientBoostClassifier:
    def __init__(self, n_estimators=100, max_leaf_nodes=8, lr=0.1, random_state=42):

        self.lr = lr
        self.max_leaf_nodes = max_leaf_nodes
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.regressors = []
        self.logodds = None
    

    def fit(self, X, y):

        n_sample = X.shape[0]
        
        # initializing p for all samples
        p = np.mean(y)

        # Initializing logodds for all samples!
        self.logodds = np.log(p/(1-p))
        y_log_odds = np.full(n_sample, self.logodds)
        y_hat = self._sigmoid(y_log_odds)

        # Training loop
        for _ in range(self.n_estimators):
            
            # Residuals
            residuals = y - y_hat

            # Although it is a classification problem, we should use 
            # regression trees since pseudo residuals are continuous values.
            reg = tree.DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, 
                                             random_state=self.random_state)
            reg.fit(X, residuals)
            assert reg.get_n_leaves() == self.max_leaf_nodes, 'Inconsistent number of leaf nodes!'
            # Adjusting tree outputs on each leaves 
            # Reference: Josh Starmer StatQuest (https://www.youtube.com/watch?v=jxuNLH5dXCs)
            leaves = reg.apply(X) # identifies a leaf index for each sample
            leaf_value = {}

            for leaf in np.unique(leaves):
                idx = (leaves==leaf)
                denomerator = np.sum(y_hat[idx]*(1-y_hat[idx]))
                denomerator = denomerator if denomerator > 1e-15 else 1e-15
                leaf_value[leaf] = np.sum(residuals[idx]) / denomerator

            # Updating logodds for each sample!
            for idx in range(n_sample):
                y_log_odds += self.lr * leaf_value[leaves[idx]]

            # Calculating new probability values for each samples
            y_hat = self._sigmoid(y_log_odds)

            # Saving regressors for prediction part
            self.regressors.append((reg, leaf_value))

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred_log_odds = np.full(n_samples, self.logodds)
    
        for reg, leaf_value in self.regressors:
            leaves = reg.apply(X)
            for idx in range(n_samples):
                y_pred_log_odds[idx] += self.lr * leaf_value[leaves[idx]]

        y_hat = self._sigmoid(y_pred_log_odds)
        return np.where(y_hat>0.5, 1, 0)

    def _sigmoid(self,x):
        x = np.clip(x, -100, 100)
        return 1/(1+np.exp(-x))
    

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier

    # Data
    X,y = make_classification(n_samples=1000, 
                              n_classes=2, 
                              n_features=5, 
                              n_informative=3, 
                              random_state=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostClassifier(n_estimators=100, max_leaf_nodes=8, lr=0.1, random_state=42)
    model.fit(X_train, y_train)
    # Predict on training data and compute accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("My Model Training Accuracy:", accuracy)
    

    # Training Sklearn GradientBoostClassifier
    skboost = GradientBoostingClassifier(n_estimators=100, 
                                         max_leaf_nodes=8, 
                                         max_depth=None,
                                         random_state=42)
    skboost.fit(X_train, y_train)

    # Predict on training data and compute accuracy
    y_pred = skboost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Scikit-Learn Training Accuracy:", accuracy)