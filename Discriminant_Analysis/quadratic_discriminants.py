import numpy as np
# Reference:  Intoroduction to statistical learning
# Link     :  https://youtu.be/oJc2r246VoQ?si=4uveXmrx0SSIQ9Kg

class QDA:
        def fit(self, X, y):
            # Dimensions
            self.n_samples, self.n_features = X.shape
            # labels in y should start from 0(e.g. class 1, 2, 3 translates to 0, 1, 2)
            self.labels = np.unique(y)
            self.n_classes  = len(self.labels)
            
            self.nk = np.array([(y==label).sum() for label in self.labels])

            self.Cik  = self._covinv_k(X, y)
            self.detC = np.array([np.linalg.det(Ck) for Ck in self.Cik])
            self.pb_k = self.nk/self.n_samples
            self.muC  = np.array([X[y==label].mean(axis=0) for label in self.labels])

        def predict(self, X_test):
            sigk  = np.zeros(self.n_classes)
            n_samples_test = X_test.shape[0]
            ypred = np.zeros(n_samples_test)

            for sample in range(n_samples_test):
                for c in self.labels:
                    xt = X_test[sample] - self.muC[c]
                    sigk[c] = -0.5 * xt.T.dot(self.Cik[c]).dot(xt) \
                                    + np.log(self.pb_k[c]) \
                                    + 0.5 * np.log(self.detC[c])
                # Prediction based on sigma values
                ypred[sample] = np.argmax(sigk)
            
            return ypred
        
        def _covinv_k(self, X, y):
            assert all(self.nk > 1) , 'All classed must contain at least 2 members!'
            # 1- calculating cov matrix for each class
            # 2- inverting each calculating matrix
            return np.array([np.linalg.inv(np.cov(X[y == label].T)) for label in self.labels])

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # Iris Dataset
    X, y = load_iris(return_X_y= True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.5, 
                                                        random_state=42, 
                                                        shuffle=True)
    
    # Fitting Model
    qda = QDA()
    qda.fit(X_train, y_train)

    # Prediction
    y_pred = qda.predict(X_test)
    print(f'My QDA Accuracy: {accuracy_score(y_pred, y_test)}')