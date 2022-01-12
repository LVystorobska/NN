import numpy as np
import matplotlib.pyplot as plt

def radial_based_func(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def err_func(y_hat, y):
    return 0.5*(y_hat - y)**2

def kmeans(X, k):
    clusters = np.random.choice(np.squeeze(X), size=k)
    prev_clusters = clusters.copy()
    converged = False

    while not converged:
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
        nearest_cluster = np.argmin(distances, axis=1)

        for i in range(k):
            pointsForCluster = X[nearest_cluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        converged = np.linalg.norm(clusters - prev_clusters) < 1e-6
        prev_clusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    nearest_cluster = np.argmin(distances, axis=1)
    # print(distances)
    # print(nearest_cluster)
    print('Converged clusters:', clusters)
    return clusters

class RBF_Model(object):
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=radial_based_func):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        self.centers = kmeans(X, self.k)
        dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                loss = (y[i] - F).flatten() ** 2

                # gradient backward
                error = -(y[i] - F).flatten()
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
            if epoch%20 == 0:
                print('Loss: {0:.4f}'.format(loss[0]))

    def predict(self, X, y):
        y_pred = []
        err_data = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            err_i = err_func(F, y[i])
            err_data.append(err_i)
            y_pred.append(F)
        print('Prediction error:',np.mean(err_data)*100)
        return np.array(y_pred)

sample_size = 200
X = np.random.uniform(0., 1.1, sample_size)
X = np.sort(X, axis=0)
y = np.sin(X)**2 

rbfnet = RBF_Model(lr=0.04, k=8, epochs=400)
rbfnet.fit(X, y)

y_pred = rbfnet.predict(X, y)

plt.plot(X, y, 'g', marker='.', label='Y-true')
plt.plot(X, y_pred, 'y',marker='1', label='RBF-prediction')
plt.legend()

plt.tight_layout()
plt.show()
