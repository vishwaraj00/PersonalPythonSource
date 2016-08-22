#!/usr/bin/python 
from __future__ import print_function
import time, pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

np.random.seed(0)
data_str = 'dataset7000.npz'
print("Dataset: "+data_str)
dataset = np.load(data_str)
data, y = (dataset['x'],dataset['y'])
data = np.asarray(data, dtype=np.float32)
n_clusters_ = 4

#clustering_names = ['MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering', 'Ward', 'AgglomerativeClustering', 'DBSCAN', 'Birch']
clustering_names = ['Ward']

#scaled_data = scale(data)  # to be used for PCA
#pca = PCA(n_components=2).fit(scaled_data) # for visualization with scaled data
#pca = PCA(n_components=2).fit(data) # for visualization, since our matrix is very sparce no need to normalize
#reduced_data = pca.transform(data)

# estimate bandwidth for mean shift
#bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(data, n_neighbors=100, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# create clustering estimators for comparision
#ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters_, n_init=20, max_iter=1000)
ward = cluster.AgglomerativeClustering(n_clusters=n_clusters_, linkage='ward', connectivity=connectivity)
#spectral = cluster.SpectralClustering(n_clusters=n_clusters_, eigen_solver='arpack', affinity="nearest_neighbors")
#dbscan = cluster.DBSCAN(eps=0.3,min_samples=50)
#affinity_propagation = cluster.AffinityPropagation(max_iter=1000, damping=.9, preference=-200,convergence_iter=30)
#average_linkage = cluster.AgglomerativeClustering(n_clusters=n_clusters_, linkage="average", affinity="l1", connectivity=connectivity)
#birch = cluster.Birch(n_clusters=n_clusters_)

#clustering_algorithms = [two_means, affinity_propagation, ms, spectral, ward, average_linkage, dbscan, birch]
clustering_algorithms = [ward]

for name, algorithm in zip(clustering_names, clustering_algorithms):
    print(79 * '_')
    print(79 * '_')
    print(algorithm)
    # predict cluster memberships
    t0 = time.time()
    algorithm.fit(data)
    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
        n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    else:
        y_pred = algorithm.predict(data)

    print(79 * '_')
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(y, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, y_pred))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y, y_pred))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y, y_pred))
