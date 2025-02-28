from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from regression_trees import RegressionTree

@dataclass
class RFRegressor:
      max_depth: Optional[int] = None
      min_samples_split: int = 2
      n_trees: Optional[int] = None
      regressors: Optional[list] = field(default_factory=list)
      n_feats: int = 2

      def fit(self, X, y):
            for _ in range(self.n_trees):
                  reg = RegressionTree(
                        n_feats=self.n_feats, 
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split)

                  Xboot, yboot = self._bootstrap(X, y)
                  reg.fit(Xboot, yboot)
                  self.regressors.append(reg)

      def predict(self, X):
            pred = np.array([reg.predict(X) for reg in self.regressors])
            y_pred = np.mean(pred, axis=0)
            return y_pred
              
      def _bootstrap(self, X, y):
            n_samples = X.shape[0]
            idx = np.random.choice(n_samples, n_samples, replace=True)
            return X[idx], y[idx]
      
###############################################################################################
###############################################################################################
if __name__ == '__main__':
      from sklearn.metrics import accuracy_score
      from sklearn.model_selection import train_test_split
      import matplotlib.pyplot as plt

      # Data
      PI = np.pi 
      num = 1000
      np.random.seed(0)
      X = np.linspace(-PI, PI+1, num=num)
      X = X.reshape(X.shape[0],1)

      y = np.sin(X) + 0.1*np.random.normal(0,1,size=(num,1))
      y = y.reshape(-1,) # y must be a one dimensional array

      X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2, 
                                                random_state=1234)

      # Model Fitting and Inference
      regr = RFRegressor(min_samples_split=5, max_depth=5, n_trees=100)
      regr.fit(X_train, y_train)
      y_pred = regr.predict(X_test)

      # Plot Model Prediction versus original data points
      plt.figure(figsize=(8,4))
      plt.scatter(X_test,y_test, color = 'blue',alpha=0.6, label='Test Data')
      plt.scatter(X_test,y_pred, color = 'black', label = 'RF Predictions')
      plt.grid()
      plt.legend()
      plt.show()