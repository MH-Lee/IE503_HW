import numpy as np
import matplotlib.pyplot as plt
from package.kmeans import  Kmeans
from package.softKmeans import SoftKmeans
from utils.make_dataset import make_dataset

## balanced and spherical
balanced_spherical_X, balanced_spherical_y = make_dataset(sample_n=10000, mode='balanced', variance='spherical')
hard_kmeans = Kmeans(K=2, opt_method='cost_fuction')
cluster, centroids = hard_kmeans.fit(balanced_spherical_X)
hard_kmeans.plot_k_means(balanced_spherical_X, cluster, centroids)

skmeans = SoftKmeans(beta=1.0, opt_method='cost_fuction')
responsibility, centroids = skmeans.fit(balanced_spherical_X)
skmeans.plot_k_means(balanced_spherical_X, responsibility, centroids)

## imbalanced and spherical
imbalanced_spherical_X, imbalanced_spherical_y = make_dataset(sample_n=10000, mode='imbalanced', variance='spherical')
hard_kmeans = Kmeans(K=2, opt_method='cost_fuction')
cluster, centroids = hard_kmeans.fit(imbalanced_spherical_X)
hard_kmeans.plot_k_means(imbalanced_spherical_X, cluster, centroids)

skmeans = SoftKmeans(beta=1.0, opt_method='cost_fuction')
responsibility, centroids = skmeans.fit(imbalanced_spherical_X)
skmeans.plot_k_means(imbalanced_spherical_X, responsibility, centroids)


## balanced and non-spherical
balanced_non_spherical_X, balanced_non_spherical_y = make_dataset(sample_n=10000, mode='balanced', variance='non-spherical')
hard_kmeans = Kmeans(K=2, opt_method='cost_fuction')
cluster, centroids = hard_kmeans.fit(balanced_non_spherical_X)
hard_kmeans.plot_k_means(balanced_non_spherical_X, cluster, centroids)

skmeans = SoftKmeans(beta=1.0, opt_method='cost_fuction')
responsibility, centroids = skmeans.fit(balanced_non_spherical_X)
skmeans.plot_k_means(balanced_non_spherical_X, responsibility, centroids)

## imbalanced and non-spherical
imbalanced_non_spherical_X, imbalanced_non_spherical_y= make_dataset(sample_n=10000, mode='imbalanced', variance='non-spherical')
hard_kmeans = Kmeans(K=2, opt_method='cost_fuction')
cluster, centroids = hard_kmeans.fit(imbalanced_non_spherical_X)
hard_kmeans.plot_k_means(imbalanced_non_spherical_X, cluster, centroids)

skmeans = SoftKmeans(beta=1.0, opt_method='cost_fuction')
responsibility, centroids = skmeans.fit(imbalanced_non_spherical_X)
skmeans.plot_k_means(imbalanced_non_spherical_X, responsibility, centroids)

# ## imbalanced and spherical
# imbalanced_spherical_X, imbalanced_spherical_y = make_dataset(sample_n=10000, mode='imbalanced', variance='spherical')
# hard_kmeans = Kmeans(K=2, opt_method='cost_fuction')
# cluster, centroids = hard_kmeans.fit(imbalanced_spherical_X)
# hard_kmeans.make_plot(imbalanced_spherical_X, cluster, centroids)
#
# skmeans = SoftKmeans(K=2, opt_method='cost_fuction')
# responsibility, centroids = skmeans.fit(imbalanced_spherical_X)
# skmeans.plot_k_means(imbalanced_spherical_X, responsibility, centroids)
#
# balanced_non_spherical_X, balanced_non_spherical_y  = make_dataset(sample_n=10000, mode='balanced', variance='non-spherical')
# hard_kmeans = Kmeans(early_stop=30, opt_method='cost_fuction')
# cluster, centroids = hard_kmeans.fit(balanced_non_spherical_X)
# hard_kmeans.make_plot(balanced_non_spherical_X, cluster, centroids)
#
# skmeans = SoftKmeans(early_stop=20)
# responsibility, centroids = skmeans.fit(balanced_non_spherical_X)
#
# skmeans.plot_k_means(balanced_non_spherical_X, responsibility, centroids)
#
# imbalanced_non_spherical_X, imbalanced_non_spherical_y = make_dataset(sample_n=5000, mode='imbalanced', variance='non-spherical')
# hard_kmeans = Kmeans(early_stop=30)
# cluster, centroids = hard_kmeans.fit(imbalanced_non_spherical_X)
# hard_kmeans.make_plot(imbalanced_non_spherical_X, cluster, centroids)
# skmeans = SoftKmeans(K=2, opt_method='distance')
# responsibility, centroids = skmeans.fit(imbalanced_non_spherical_X)
# skmeans.plot_k_means(imbalanced_non_spherical_X, responsibility, centroids)
