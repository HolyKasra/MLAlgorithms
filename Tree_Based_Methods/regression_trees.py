from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    samples: Optional[int] = None
    mse_loss: Optional[float] = None

@dataclass
class RegressionTree:
    n_feats: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    root: Optional[Node] = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.labels = np.unique(y)
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self, X, y, depth=0):

        n_samples, _ = X.shape
        
        best_feature, best_threshold, best_mse = self._best_split(X, y)
        current_mse = self._mse(y)

        # Base cases for the recursive algorithm
        if (self.max_depth is not None and depth >=self.max_depth) or \
           (n_samples <= self.min_samples_split):
            
            leaf_value = np.mean(y)
            return Node(value=leaf_value, samples=n_samples, mse_loss=current_mse)
        
        if best_feature is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value, samples=n_samples, mse_loss=current_mse)
        

        best_left = X[:, best_feature] <= best_threshold
        best_right= ~best_left


        left  = self._grow_tree(X[best_left], y[best_left], depth=depth+1)
        right = self._grow_tree(X[best_right], y[best_right], depth=depth+1)


        return Node(feature=best_feature, threshold=best_threshold,
                    left=left, right=right, samples=n_samples, mse_loss=current_mse)

    def _best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_MSE = float('inf')

        n_features = X.shape[1]
        featidxs = np.random.choice(n_features, self.n_feats, replace=False)

        MSE_parent = self._mse(y)

        for featidx in featidxs:
            thresholds = self._thresh(np.unique(X[:, featidx]))
            for thresh in thresholds:
                left_side = X[:, featidx] <=thresh
                right_side = ~left_side

                nL = np.sum(left_side)
                nR = np.sum(right_side)

                if nL==0 or nR==0:
                    continue
                
                
                y_left = y[left_side]
                y_right = y[right_side]
                
                
                MSE_left = self._mse(y_left)
                MSE_right = self._mse(y_right)
                MSE_weighted = (nL * MSE_left  + nR * MSE_right)/len(y)

                MSE_gain = MSE_parent - MSE_weighted
                

                if MSE_weighted < best_MSE:
                    best_feature = featidx
                    best_threshold = thresh
                    best_MSE = MSE_weighted
        
        return best_feature, best_threshold, best_MSE

    def _mse(self, data):
        mu = data.mean()
        return np.mean((data-mu)**2)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X]) 

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        
        return self._predict(x, node.right)
    

    def _thresh(self, x):
        return np.array([0.5 * x[i] + 0.5 * x[i+1] for i in range(len(x)-1)])
    
if __name__ == '__main__':
    from sklearn import tree
    import matplotlib.pyplot as plt

    # Datas
    PI = np.pi 
    num = 100
    np.random.seed(0)
    X = np.linspace(-PI, PI+1, num=num)
    X = X.reshape(X.shape[0],1)

    y = np.sin(X) + 0.1*np.random.normal(0,1,size=num).reshape(X.shape[0],1)

    reg_tree = RegressionTree(min_samples_split=10, max_depth=5)
    reg_tree.fit(X, y)
    y_pred_custom = reg_tree.predict(X)

    # Training Regression Tree with Scikit-Learn to compare
    reg_sk = tree.DecisionTreeRegressor(min_samples_split=10, max_depth=5)
    reg_sk.fit(X,y)
    y_pred_sk = reg_sk.predict(X)

    plt.style.use('ggplot')

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.scatter(X,y, color = 'blue',alpha=0.6, label='Original Data')
    plt.plot(X,y_pred_sk, color = 'red',label='ScikitLearn Tree')
    plt.legend()


    plt.subplot(122)
    plt.scatter(X,y, color = 'blue',alpha=0.6, label='Original Data')
    plt.plot(X,y_pred_custom, color = 'red',label='My Tree')
    plt.legend()
    plt.show()
