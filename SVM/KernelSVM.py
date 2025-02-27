import numpy as np

class KernelSVC:
    def __init__(self, lr=0.01, C=1, kernel='rbf', gamma=1, max_iter=1000, tol=1e-3):
        self.lr = lr
        self.C = C
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.grad = None
        self.K = None
        self.b = 0
        self.X_sv = None
        self.y_sv = None
        self.alpha_sv = None

    def _kernel(self, x, y):
        match self.kernel:
            case 'rbf':
                return np.exp(-self.gamma * np.sum((x - y) ** 2))
            case 'linear':
                return np.dot(x, y)
            case 'poly':
                return (np.dot(x, y) + 1) ** self.gamma

    def _kernel_mat(self, X):
        n_sample = X.shape[0]
        K = np.zeros(shape=(n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                K[i, j] = self._kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)

        # Compute kernel matrix
        self.K = self._kernel_mat(X)

        loss_val = float('inf')

        for _ in range(self.max_iter):
            self.grad = np.zeros(n_samples)
            for i in range(n_samples):
                self.grad[i] = 1 - y[i] * np.sum(self.alpha * y * self.K[:,i])
                
            # Update alpha
            self.alpha += self.lr * self.grad
            self.alpha = np.clip(self.alpha, 0, self.C)

            # Compute loss
            loss = self._loss(X, y)

            # Check for convergence
            if np.abs(loss - loss_val) < self.tol:
                break
            # Updating Best Loss
            loss_val = loss

        # Compute bias term
        support_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        if len(support_indices) > 0:
            for s in support_indices:
                self.b += y[s] - np.sum(self.alpha * y * self.K[s, :])
            self.b /= len(support_indices)

        # Store support vectors
        sv_indices = np.where((self.alpha > 1e-5))[0]
        self.X_sv = X[sv_indices]
        self.y_sv = y[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]

    def predict(self, X):
        n_samples = X.shape[0]
        decision = np.zeros(n_samples)

        # Founding decision for every test data sample!
        for i in range(n_samples):
            decision[i] = np.sum([
                alpha * y_sv * self._kernel(X[i], x_sv) 
                for alpha, y_sv, x_sv in zip(
                    self.alpha_sv, 
                    self.y_sv, 
                    self.X_sv
                )
            ]) + self.b
        
        return np.sign(decision)

    def _loss(self, X, y):
        n_samples = X.shape[0]
        loss = 0
        for i in range(n_samples):
            loss += self.alpha[i]
            for j in range(n_samples):
                loss -= 0.5 * self.alpha[j] * self.alpha[i] * y[i] * y[j] * self.K[i, j]
        return loss
    
if __name__ == '__main__':
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from decision_boundary import plot_decision_boundary

    # Data
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train My SVM with RBF kernel
    svm = KernelSVC(kernel='rbf', C=1.0, gamma=1.0, max_iter=300)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"My SVC Accuracy: {accuracy*100}%")


    # Train Scikit-Learn SVM with RBF kernel
    svm = SVC(kernel='rbf', C=1.0, gamma=1.0, max_iter=300)
    svm.fit(X_train, y_train)

    # Make predictions
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Scikit-Learn SVC Accuracy: {accuracy*100}%")

    # Plot Decision Boundary Of My Model
    plot_decision_boundary(svm, X, y)