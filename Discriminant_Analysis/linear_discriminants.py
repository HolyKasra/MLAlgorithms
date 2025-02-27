import numpy as np

class LDA:

    def __init__(self, n_component):
        self.n_component = n_component

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.n_classes  = self.labels.shape[0]
        self.n_samples, self.n_features = X.shape
        self.nk = np.array([(y==label).sum() for label in self.labels])

        # Calculating the parameters
        self.pb_k    = self.nk/self.n_samples
        self.muC     = np.array([X[y==label].mean(axis=0) for label in self.labels])
        self.covinv  = self._pooled_invcov(X, y)
        self.eigvec  = self._linear_discriminants(X, y)

    def predict(self, X_test):
        sigk = np.zeros(self.n_classes)
        n_sample_test = X_test.shape[0]
        ypred = np.zeros(n_sample_test)

        for sample in range(n_sample_test):
            for c in self.labels:
                xt = X_test[sample].dot(self.covinv).dot(self.muC[c])
                sigk[c] =   xt - 0.5 * self.muC[c].T.dot(self.covinv).dot(self.muC[c]) + np.log(self.pb_k[c])
            ypred[sample] = np.argmax(sigk)
        return ypred
    
    def transform(self, X_test):
        self.X_test_lda  = X_test.dot(self.eigvec)
        return {'X_lda': self.X_test_lda, 'eigenvec': self.eigvec}

    def _linear_discriminants(self, X, y):
        # Calculating the linear discriminants!
        Sw = np.zeros((self.n_features, self.n_features))
        Sb = np.zeros((self.n_features, self.n_features))

        # calculating each class means and the total mean!s
        muT = X.mean(axis = 0)
        muC = self.muC

        # Calculating Sb, Sw
        for label in self.labels:
            for x in X[y==label]:
                Sw += np.outer(x - muC[label], (x - muC[label]).T)
            
            mu_diff = muC[label] - muT
            Sb += self.nk[label] * np.outer(mu_diff, mu_diff.T)

        # finding linear discriminants using eigenvalue/eigenvectors of inv(Sw) * Sb
        self.eigval, self.eigvec  = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        sort_idx    = np.argsort(self.eigval)[::-1]
        self.eigval = self.eigval[sort_idx]
        self.eigvec = self.eigvec[:,sort_idx]
        
        # eigenvectors are the linear discriminants!
        # choosing significant LDs (= n_components)
        return self.eigvec[:, :self.n_component] 
    
    def _pooled_invcov(self, X, y):
        assert all(self.nk >1) , 'All classed must contain at least 2 members!'
        # search for empirical covariance in sklearn!
        covmats = np.array([np.cov(X[y==label].T) for label in self.labels])
        pooled = np.zeros_like(covmats[0])

        for label in range(1, self.n_classes+1):
            pooled += ((self.nk[label-1]- 1)/(self.n_samples - self.n_classes)) * covmats[label-1]
        pooled += np.eye(self.n_features) * 1e-5
        return np.linalg.inv(pooled)
    

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    from plot_lda import plot_linear_discriminants
    
    # Iris Dataset
    X, y = load_iris(return_X_y= True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)

    # Fitting Model
    lda = LDA(n_component=2)
    lda.fit(X,y)

    # LDA as a classifier
    y_pred = lda.predict(X_test)
    print(f'My Model Accuracy: {accuracy_score(y_test, y_pred)*100:.3f}%')

    # Linear Discriminants
    info = lda.transform(X)
    X_lda = info['X_lda']
    V     = info['eigenvec']

    # Plot linear discriminants
    title = 'LDA: Iris projection onto the first 2 linear discriminants'
    plot_linear_discriminants(X_lda, y,figsize=(6,3), title=title)