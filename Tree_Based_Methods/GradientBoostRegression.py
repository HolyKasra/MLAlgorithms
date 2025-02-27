import numpy as np
from sklearn import tree

class GradientBoostRegressor:
    def __init__(self, n_estimators=10, max_leaf_nodes=8, lr=0.1, tol=1e-4):

        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.lr = lr
        self.tol = tol
        self.regressors = []
        self.loss = []
        self.mean_value = None
        self.flag = False

    def fit(self, X, y, verbose=False):

        n_samples = X.shape[0]
        # Initialize y_pred with the mean value
        self.mean_value = y.mean()
        y_pred = np.ones(y.shape) * self.mean_value

        for idx in range(self.n_estimators):
            reg = tree.DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes)
            residuals = y - y_pred
            reg.fit(X, residuals)

            assert reg.get_n_leaves() == self.max_leaf_nodes, 'Number of leaves for trees are inconsistent!'

            res_pred = reg.predict(X)
            y_pred += self.lr * res_pred

            loss_val = self._loss(y, y_pred)

            self.regressors.append(reg)
            self.loss.append(loss_val)
            
            if loss_val < self.tol:
                self.flag = True
                break

            if verbose and (idx + 1) % int(self.n_estimators/10) == 0:
                print(f"[Iteration] {idx + 1:5} ===> [Loss]  {loss_val:7.15f}")

        if verbose and self.flag:
            print(f"[FINISHED!] {idx + 1:5} ===> [Loss]  {loss_val:7.15f}")

    def predict(self, X):
        prediction = np.ones(X.shape[0]) * self.mean_value
        for regressor in self.regressors:
            prediction += self.lr * regressor.predict(X)
        return prediction
    
    @property
    def n_estimators_used(self):
        return len(self.regressors)

    def _loss(self, y_true, y_pred):
        # Example loss function: Mean Squared Error (MSE)
        return np.mean((y_true - y_pred) ** 2)

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X,y = make_regression(n_samples=1000, n_features=20, n_informative=10, random_state=2025)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # With verbose
    model = GradientBoostRegressor(n_estimators=200, max_leaf_nodes=8, lr=0.1, tol=1e-4)
    model.fit(X_train, y_train, verbose=True)

    y_pred = model.predict(X_test)
    print('=======================================================')
    print(f'My GradientBoostRegressor R_square: {r2_score(y_test, y_pred):.4f}')