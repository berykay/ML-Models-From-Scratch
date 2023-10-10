import random

class KMeansClustering():
    def __init__(self, n_cluster=3,  max_iterations=200):
        self.n_cluster = n_cluster
        self.max_iterations = max_iterations

    def fit(self, X):
        self.n_samples, self.n_features = len(X), len(X[0])
        #self.centroids = [X[i] for i in random.sample(range(self.n_samples), self.n_cluster)]
        self.centroids = self.randomUniform(self.aMin(X), self.aMax(X), self.n_cluster)
        for _ in range(self.max_iterations):
            self.clusters = [[] for _ in range(self.n_cluster)]

            for data_points in X:
                closest_centroid = self.closestCentroid(data_points)
                self.clusters[closest_centroid].append(data_points)

            for i in range(self.n_cluster):
                self.centroids[i] = self.calculateCentroid(self.clusters[i])
        return self.centroids, self.clusters
    
    def sumOfSquaredDistances(self, X, predictions):
        result = 0
        for i in range(len(X)):
            result += self.euclideanDistance(X[i],self.centroids[predictions[i]])
        return result

    def predict(self, X):
        y_pred = [self.closest(x) for x in X]
        return y_pred
    
    def closest(self, x):
        closest_i = 0
        closest_dist = float("inf")
        for i in range(self.n_cluster):
            dist = self.euclideanDistance(x, self.centroids[i])
            if dist < closest_dist:
                closest_i = i
                closest_dist = dist
        return closest_i

    def closestCentroid(self, data_points):
        closest_centroid = 0
        for i in range(self.n_cluster):
            if self.euclideanDistance(data_points, self.centroids[i]) < self.euclideanDistance(data_points, self.centroids[closest_centroid]):
                closest_centroid = i
        return closest_centroid
    
    def calculateCentroid(self, cluster):
        centroid = [0] * self.n_features
        for clstr in cluster:
            for i in range(self.n_features):
                centroid[i] += clstr[i]
        if len(cluster) > 0:
            for i in range(self.n_features):
                centroid[i] /= len(cluster)
        return centroid

    def euclideanDistance(self, data_points, centroids):
        return sum([(data_points[i] - centroids[i])**2 for i in range(len(data_points))])**0.5

    def aMin(self, X):
        min = [float('inf')] * len(X[0])  
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] < min[j]:
                    min[j] = X[i][j]
        return min
    
    def aMax(self, X):
        max = [float('-inf')] * len(X[0])  
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i][j] > max[j]:
                    max[j] = X[i][j]
        return max
        
    def randomUniform(self, min, max, size):
        centroids = []
        for _ in range(size):
            centroid = [random.uniform(min[j], max[j]) for j in range(len(min))]
            centroids.append(centroid)
        return centroids
    