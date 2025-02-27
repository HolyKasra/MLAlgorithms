import numpy as np
import collections
####################################################################################################################
####################################################################################################################
# KNN and Weighted KNN Models
class KNNClassifier:
    def __init__(self, K=1, method='euclidean'):
        self._K = K
        self._method = method

    def fit(self, X, y):
        self.X = X 
        self.y = y

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
        dst = np.array([self.dist(x, xtrain, method=self._method) for xtrain in self.X])
        idmax = np.argsort(dst)[:self._K]
        votes_count = collections.Counter(self.y[idmax])
        return votes_count.most_common(1)[0][0]

    @staticmethod
    def dist(x, y, method):
        if method == 'euclidean':
            return np.linalg.norm(x-y)

        elif method =='manhattan':
            return np.abs(x-y).sum()
####################################################################################################################
####################################################################################################################        
class WeightedKNN:
    def __init__(self, n_neighbors=2, **params):
        self.n_neighbors = n_neighbors

        if 'method' not in params:
            raise ValueError("The 'method' parameter must be provided!")
        
        self.method = params['method']

        if self.method == 'gaussian':
            if 'sigma' not in params:
                raise KeyError("The 'sigma' parameter must be provided for the 'gaussian' method!")
            self.sigma = params['sigma']
        elif self.method != 'inverse':
            raise KeyError('Please Enter a valid key!')

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X]) 

    def _predict(self, x):
        # Features are on columns
        dist = np.linalg.norm(x-self.X, axis=1)
        idx = dist.argsort()[:self.n_neighbors]
        weights = self._weights(dist=dist)[idx]

        # Extracting labels in the neighborhood
        neighbors_labels = self.y[idx]
        
        # Sum weights per class; predict the class with the highest total weight
        vote = []
        for label in np.unique(neighbors_labels):
            weight_sum = np.sum(weights[neighbors_labels==label])
            vote.append(weight_sum)
        
        return np.array(vote).argsort()[-1]

    def _weights(self, dist, epsilon=1e-10):
        match(self.method):
            case 'inverse':
                return 1/(dist+epsilon)
            
            case 'gaussian':
                return np.exp(-dist**2/self.sigma**2)
####################################################################################################################  
#############################################  KNN   ###############################################################         
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # Data
    X,y = make_classification(
                            n_samples=150, 
                            n_classes=2, 
                            n_features=2, 
                            n_informative=2,
                            n_redundant=0, 
                            n_clusters_per_class=1, 
                            random_state=123
                            )

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        random_state=42, 
                                                        shuffle=True)

    # Fitting
    K=3
    knn = KNNClassifier(K=K)
    knn.fit(X_train, y_train)

    # Prediction
    y_pred = knn.predict(X_test)
    print(f'Accuracy of KNN for K={K} : {accuracy_score(y_test, y_pred)*100}%')