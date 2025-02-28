import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, random_state=1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels = None
        self.centers = None

    def fit(self, X):

        rng = np.random.RandomState(seed=self.random_state)
        centers = rng.permutation(X)[:self.n_clusters]
        converged = False

        while not converged:
            dist = {key:[] for key in range(self.n_clusters)}
            for x in X:
                distance = np.linalg.norm(x-centers, axis=1)
                idx = np.argmin(distance)
                dist[idx].append(x)
            
            updated_centers = np.array([np.mean(dist[c], axis=0) for c in range(self.n_clusters)])
            converged = np.allclose(updated_centers, centers)
            centers = updated_centers.copy()
        
        self.centers = centers

    def assign_labels(self, X):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for c in range(self.n_clusters):
            distance[:, c] = np.linalg.norm(self.centers[c]-X, axis=1)
        return np.argmin(distance, axis=1)
    
    def plot_clusters(self, X, title=None, alpha=0.6):
        centers = self.centers
        labels = self.assign_labels(X)
        # Plot Clusters
        plt.figure(figsize=(4,2))
        plt.style.use('ggplot')
        plt.scatter(X[:,0],X[:,1],c=labels, alpha=alpha)
        if centers is not None:
            plt.scatter(centers[:,0], centers[:,1], s=100, color='r', marker='x')
        plt.title(title)
        plt.show()   

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, centers=4, random_state=2025)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    y_labels = kmeans.assign_labels(X)
    kmeans.plot_clusters(X, title='K-Means Clustering Result')