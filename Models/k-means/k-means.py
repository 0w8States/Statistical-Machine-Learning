import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the samples
data = np.load('AllSamples.npy')

# Create the initial points
k1 = 3
i_point1 = np.array([[6.03237178, 8.86195452], [5.02471033, 8.23879873], [3.89523379, 0.70718356]])
k2 = 5
i_point2 = np.array([[2.80096609, 1.03176348], [5.57009665, 8.3870942], [1.713841, 4.31350258], [7.35456962, 0.93930822], [5.33498937, 3.07430754]])

print("Initial Point Data")
print(k1)
print(i_point1)
print(k2)
print(i_point2)


# Plot the initial points, just for a starting view of the data
sns.scatterplot(data[:,0], data[:, 1])
sns.scatterplot(i_point1[:,0], i_point1[:, 1], s=100, color='y')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

class KMeans():
    def __init__(self):
        self.prev_loss = 0

    def pairwise_dist(self, x, y):
        xSumSquare = np.sum(np.square(x), axis=1);
        ySumSquare = np.sum(np.square(y), axis=1);
        mul = np.dot(x, y.T);
        dists = np.sqrt(abs(xSumSquare[:, np.newaxis] + ySumSquare - 2 * mul))
        return dists

    def _update_assignment(self, centers, points):
        row, col = points.shape
        cluster_idx = np.empty([row])
        distances = self.pairwise_dist(points, centers)
        cluster_idx = np.argmin(distances, axis=1)
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):
        K, D = old_centers.shape
        new_centers = np.empty(old_centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(points[cluster_idx == i], axis=0)
        return new_centers

    def _get_loss(self, centers, cluster_idx, points):
        dists = self.pairwise_dist(points, centers)
        loss = 0.0
        N, D = points.shape
        for i in range(N):
            loss = loss + np.square(dists[i][cluster_idx[i]])

        return loss

    # Run through the itterations and fit
    def fit(self, K, centers, data, max_ittr=300, abs_tol=0, rel_tol=0):
        # First step is to initialize the centers of the centeriods, in this case they are given
        for ittr in range(max_ittr):
            cluster_idx = self._update_assignment(centers, data)
            centers = self._update_centers(centers, cluster_idx, data)
            loss = self._get_loss(centers, cluster_idx, data)
            K = centers.shape[0]
            if ittr:
                diff = np.abs(self.prev_loss - loss)
                if diff < abs_tol and diff / self.prev_loss < rel_tol:
                    break
                self.prev_loss = loss
        return cluster_idx, centers, loss


# Create a new KMeans model
kmeans = KMeans()
# Fit the model
cluster_idx, centers, loss = kmeans.fit(k1, i_point1, data)

print("\nClustering")
print(cluster_idx)
print("Loss")
print(loss)
print("Centers")
print(centers)

# Plot the final clusters and datapoints
sns.scatterplot(data[:,0], data[:, 1], hue=cluster_idx)
sns.scatterplot(centers[:,0], centers[:, 1], s=100, color='y')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()