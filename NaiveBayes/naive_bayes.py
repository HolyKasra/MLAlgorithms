import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, self.n_features = X.shape
        self.labels = np.unique(y)
        self.n_classes  = len(self.labels)
        
        nk = np.array([(y==label).sum() for label in self.labels])
        self.pb_k = nk / n_samples

        self.muC = np.array([X[y==c].mean(axis = 0) for c in self.labels])
        self.varC = np.array([X[y==c].var(axis = 0) for c in self.labels])

    # it is just a wrapper around the main _predict()!
    def predict(self, X_test):
        y_pred = np.array([self._predict(x_test) for x_test in X_test])
        return y_pred

    # interesting approach for prediction
    # this method makes sure that the user does not access to the implemented logic of predict!
    def _predict(self, x_test):
        sigk = np.zeros(self.n_classes)
        for label in range(self.n_classes):
            sigk[label] = np.sum(self._logpdf(x_test, label)) + np.log(self.pb_k[label])
        return self.labels[np.argmax(sigk)]

    def _logpdf(self, x, idx):
        numerator = np.exp(-((x - self.muC[idx]) ** 2) / (2 * self.varC[idx]))
        denominator = np.sqrt(2 * np.pi * self.varC[idx])
        return np.log(numerator / denominator)

    

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # Dataset
    X, y = make_classification(n_samples=1000, 
                               n_features=3, 
                               n_informative=3, 
                               n_redundant=0,
                               n_classes=3,
                               n_clusters_per_class=1, 
                               random_state=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=123)
    # Trainign Model
    nb = NaiveBayes()
    nb.fit(X_train,y_train)

    # Prediction
    y_pred = nb.predict(X_test)
    print(f'My Naive Bayes Accuracy: {accuracy_score(y_test, y_pred)*100}%')