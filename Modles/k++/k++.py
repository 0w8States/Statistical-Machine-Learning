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


# Plot the initial points, just for a starting view of the data
sns.scatterplot(data[:,0], data[:, 1])
sns.scatterplot(i_point1[:,0], i_point1[:, 1], s=100, color='y')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# Class implementation of the K-means algorithm
class KMeans():

  # Constructor
  def __init__(self, K, centroid_initializer = "default", initial_points = np.empty((0, 2)), verbose = False, plotter=False):
    self.K = K
    self.initializer = centroid_initializer
    self.initial_points = initial_points
    self.centroids = self.initial_points
    self.verbose = verbose
    self.plotter = plotter
    self.loss = 0
    self.cluster_idx = []

    # Log outputs if in debug mode
    self.__logger(f'Number of Centroids: {self.K}')
    self.__logger(f'Centroids Initializer: {self.initializer}')
    self.__logger(f'Initial Centroids:\n{self.initial_points}\n\n')

  # Cacluate the full Euclidean Distance
  def __euclidean_distance(self, point1, point2):
    return self.squared_distance(point1, point2)**0.5

  # Cacluate the Euclidean Distance Squared
  def __squared_distance(self, point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

  # Calcuate the distances over a numpy array
  def __calc_distances(self, centroids, points):
      # Take a look at the geeks for geek on using the linalg
      centSumSquare = np.sum(np.square(centroids),axis=1);
      pointsSumSquare = np.sum(np.square(points),axis=1);
      mul = np.dot(centroids, points.T);
      distances = np.sqrt(abs(centSumSquare[:, np.newaxis] + pointsSumSquare-2*mul))
      return distances

  # K++ Initializer - Takes an initial point from the dataset
  #                   and calculated all the centroids based on
  #                   maximum average distance.
  def __kpp_init(self, points):
      centroids = np.empty((0,2))
      centroids = np.append(centroids, self.initial_points, axis=0)
      # Iterate until all Ks are found
      for ittr_k in range(self.K-1):
        # All distances to each centroid
        total_distances = []
        # Average Distance a point is from all centroids
        avg_distance = []
        # Calculate all the distances to each centroid
        for centroid in centroids:

          # Individual run of distances to a specific centroid
          distances = []
          for point in points:
            distances.append(self.__squared_distance(point, centroid))
          total_distances.append(distances)

        avg_distance = total_distances[0]

        # Sum the distances accross all centroids
        if len(total_distances) > 1:
          for ittr_c in range(1,len(total_distances)):
            for ittr_p in range(len(total_distances[ittr_c])):
              avg_distance[ittr_p] = avg_distance[ittr_p] + total_distances[ittr_c][ittr_p]

        # Find the average of all the distances
        for ittr_p in range(len(avg_distance)):
          avg_distance[ittr_p] = avg_distance[ittr_p] / len(centroids)

        # Locate the next point via maximum average distance
        max_value = max(avg_distance)
        new_centroid_idx = avg_distance.index(max_value)
        # Add it to the centroid array
        centroids = np.append(centroids, [points[new_centroid_idx]], axis=0)
        # Remove it from the available points list
        points = np.delete(points, new_centroid_idx, axis=0)

      # Plot the data if plotter is enabled
      if self.plotter:
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        sns.scatterplot(points[:,0], points[:, 1], s=100)
        sns.scatterplot(centroids[:,0], centroids[:, 1], s=180, color='y')
        plt.title('Initializer')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

      # return the centroids found by K++
      return centroids

  # Update the centroid assignments for the points
  def __update_centroid_assignment(self, centroids, points):
        row, col = points.shape
        cluster_idx = np.empty([row])
        distances = self.__calc_distances(points, centroids)
        cluster_idx = np.argmin(distances, axis=1)
        return cluster_idx

  # Update the centroid locations
  def __update_centroids(self, old_centroids, cluster_idx, points):
        K, D = old_centroids.shape
        new_centroids = np.empty(old_centroids.shape)
        for i in range(K):
            new_centroids[i] = np.mean(points[cluster_idx == i], axis = 0)
        return new_centroids

  # Calcuate the loss function
  def __get_loss(self, centroids, cluster_idx, points):
        dists = self.__calc_distances(points, centroids)
        loss = 0.0
        N, D = points.shape
        for i in range(N):
            loss = loss + np.square(dists[i][cluster_idx[i]])
        return loss


  # Fit the model
  def fit(self, points, max_itterations=20):
    prev_loss = 0
    rel_tol = 0
    abs_tol = 0

    losses = []

    # Use the K++ Algorithm if specified, otherwise use the provided points
    if self.initializer == "k++":
      self.centroids = self.__kpp_init(points)

    # First step is to initialize the centers of the centeriods, in this case they are given
    for ittr in range(max_itterations):
      # Update the cluster indexs for each point
      self.cluster_idx = self.__update_centroid_assignment(self.centroids, points)
      # Update the centroids
      self.centroids = self.__update_centroids(self.centroids, self.cluster_idx, points)
      # Calcualte the loss function
      self.loss = self.__get_loss(self.centroids, self.cluster_idx, points)
      # Store the loss data for plotting later
      losses.append(self.loss)
      self.K = self.centroids.shape[0]

      # Check if it should stop the loop and minimum loss has been found
      if ittr:
        diff = np.abs(prev_loss - self.loss)
        if diff < abs_tol and diff / prev_loss < rel_tol:
          break
        prev_loss = self.loss

    if self.plotter:
      # Plot the final centroids and clusters
      fig = plt.gcf()
      fig.set_size_inches(16, 8)
      sns.scatterplot(points[:,0], points[:, 1], s=100, hue=self.cluster_idx)
      sns.scatterplot(self.centroids[:,0], self.centroids[:, 1], s=180, color='y')
      plt.title("Fitted")
      plt.xlabel('x1')
      plt.ylabel('x2')
      plt.show()
      # Plot the losses
      fig = plt.gcf()
      fig.set_size_inches(16, 8)
      plt.title('Losses')
      plt.plot(losses)
      plt.show()


  # Getter method for centroids
  def get_centroids(self):
    return self.centroids

  # Getter for the loss
  def get_loss(self):
    return self.loss


  # Output logger for debug purposes
  def __logger(self, log):
    if self.verbose:
      print(log)



# Use the provided 3 centroid points to complete the k-means
kmeans = KMeans(k1,initial_points = i_point1, verbose = True)
kmeans.fit(data)

print(f'Final Centroids:\n{kmeans.get_centroids()}\n')
print(f'Final Loss:\n{kmeans.get_loss()}\n')

# Use the provided 5 centroid points to complete the k-means
kmeans = KMeans(k2,initial_points = i_point2, verbose = True)
kmeans.fit(data)

print(f'Final Centroids:\n{kmeans.get_centroids()}\n')
print(f'Final Loss:\n{kmeans.get_loss()}\n')



# Stategy 2 gives you initial starting points and you use K++ to complete
i_point1 = np.array([[6.12393256, 5.49223251]])
k1 = 4
i_point2 = np.array([[5.60001917, 3.02332715]])
k2 = 6


kmeans = KMeans(k1,initial_points = i_point1, centroid_initializer = "k++", verbose = True, plotter=True)
kmeans.fit(data)
print(f'Final Centroids:\n{kmeans.get_centroids()}\n')
print(f'Final Loss:\n{kmeans.get_loss()}\n')


kmeans = KMeans(k2,initial_points = i_point2, centroid_initializer = "k++", verbose = True, plotter=True)
kmeans.fit(data)
print(f'Final Centroids:\n{kmeans.get_centroids()}\n')
print(f'Final Loss:\n{kmeans.get_loss()}\n')