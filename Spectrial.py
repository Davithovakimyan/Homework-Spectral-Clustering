import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



class SpectralClustering:

    def __init__(self, clusters=3, gamma=1):
        self.clusters = clusters
        self.gamma = gamma

    def fit(self, x):
        K = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                distance = np.linalg.norm(x[i] - x[j])
                K[i][j] = np.exp(-self.gamma * distance ** 2)

        diag = np.diag(np.sum(K, axis=1))
        D_inv = np.linalg.inv(np.sqrt(diag))
        laplacian=diag-K
        L_norm = np.identity(K.shape[0]) - np.matmul(np.matmul(D_inv, K),D_inv)


        eigenvalues, eigenvectors = np.linalg.eig(L_norm)
        index = np.argsort(eigenvalues)[1:self.clusters+1]
        self.eigenvector = eigenvectors[:, index]
        prediction = self.predict(self.eigenvector)
        plt.scatter(self.eigenvector[:, 0], self.eigenvector[:, 1], c=prediction)
        plt.show()
        return prediction

    def predict(self, x):
        model = KMeans(n_clusters=3)
        prediction = model.fit_predict(x)
        return prediction


x, y = make_blobs(centers=3, n_features=5, cluster_std=1)
cluster = SpectralClustering(clusters=3)
pred = cluster.fit(x)

print(y)
print(pred)

