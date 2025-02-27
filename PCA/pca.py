import numpy as np

class PCA:
    '''
    Simple Implementation of PCA algorithm for dimension reduction
    '''
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.__eigvals = None
        self.__eigvecs = None
        self.eigvecs_pca = None
        self.eigvals_pca = None

    
    def fit(self, X):

        # Default Mode for number of components
        if self.n_components is None:
            self.n_components = X.shape[1]

        # Covariance Matrix
        Cx = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eig(Cx)

        # Sorting eigenvectors cols with respect to related eigenvalues!
        sort_idx = np.argsort(eigvals)[::-1]
        self.__eigvecs = eigvecs[:,sort_idx]
        self.__eigvals = eigvals[sort_idx]

        # Choosing n_components as it is given!
        self.eigvecs_pca   = self.__eigvecs[:,:self.n_components]
        self.eigvals_pca   = self.__eigvals[:self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X_test):
        return X_test.dot(self.eigvecs_pca)

    @property
    def explained_variance_(self, normalized = True):
        if normalized:
            return self.__eigvals/np.sum(self.__eigvals)
        else:
            return self.__eigvals 
    @property
    def components_(self):
        return self.eigvecs_pca
    
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from plot_pca import plot_pca_iris
    
    X, y = load_iris(return_X_y= True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    title = 'PCA: Iris projection onto the first 2 Principal Components'
    plot_pca_iris(X_pca, y, title=title)
