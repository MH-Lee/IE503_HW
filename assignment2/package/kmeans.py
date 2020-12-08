import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
K : # of cluster
max_iter : set maximum iteration
early_stop : early stop when centroid dose not move
opt_method : cost_fuction(distortion_measure) or distance(distance between k and k-1 centrodids)
initial_mode : random(randomly pick data point ), image (initialize centroid within RGB range(0 ~ 255))
"""

class Kmeans(object):

    def __init__(self, K=2, early_stop=10, max_iter=500, \
                 initial_mode='random',  opt_method='cost_fuction'):
        self.K = K
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.initial_mode = initial_mode
        self.opt_method = opt_method
        self.history = list()
        print("initialize mode : {}, optimize method : {}, max_iter : {}".format(self.initial_mode, self.opt_method, self.max_iter))

    def initialize_centroid(self, X=None):
        if self.initial_mode == 'image':
            initialize_centroid = np.random.randint(0,255, (self.K, 3))
        else:
            if X is None:
                raise ValueError(f"Input X data!")
            idx_list = np.random.choice(range(X.shape[0]), self.K, replace=False)
            initialize_centroid = X[idx_list]
        return initialize_centroid

    def assign_step(self, X, centroids):
        N, D = X.shape
        euclidean_dist = np.zeros((N, self.K))
        for idx, center in enumerate(centroids):
            euclidean_dist[:,idx] = np.linalg.norm(X - center, 2, axis=1)
        cluster = np.argmin(euclidean_dist, axis=1)
        return cluster, euclidean_dist

    def update_step(self, X, cluster):
        N, D = X.shape
        new_centroids = np.zeros((self.K, D))
        for c in np.unique(cluster):
             new_centroids[c] = np.mean(X[np.where(cluster == c)], axis=0)
        return new_centroids

    def distortion_measure(self, distance, cluster):
        responsibility = np.zeros((len(cluster), self.K))
        for idx, c in enumerate(cluster):
            responsibility[idx, c] = 1
        distortion_measure = (responsibility * distance).sum()
        return distortion_measure.sum()

    def fit(self, X):
        if self.initial_mode == 'image':
            centroids = self.initialize_centroid()/255
        else:
            centroids = self.initialize_centroid(X)
        print("intiial centroids : ", centroids)
        num = centroids.shape[0]
        old_dm = float('inf') if self.opt_method == 'cost_fuction' else centroids
        early_stop_list = [0]
        iteration = 0
        self.history.append(old_dm)
        while sum(early_stop_list) < self.early_stop:
            if iteration > len(early_stop_list):
                break
            cluster, euclidean_dist = self.assign_step(X, centroids)
            centroids = self.update_step(X, cluster)
            if self.opt_method == 'cost_fuction':
                new_dm = self.distortion_measure(euclidean_dist, cluster)
                early_stop_criterion = old_dm - new_dm
                if old_dm < new_dm:
                    break
            elif self.opt_method == 'distance':
                new_dm = centroids
                early_stop_criterion = np.sqrt(np.linalg.norm(old_dm - new_dm, 2, axis=1).mean())
            else:
                raise ValueError(f"Select cost_fuction or distance")
            if iteration % 20 == 0:
                print("epoch : {}, cost change{}".format(iteration, early_stop_criterion))
            if abs(early_stop_criterion) <= 1e-3:
                early_stop_list.append(1)
            else:
                early_stop_list.append(0)
            old_dm = new_dm
            self.history.append(old_dm)
            iteration += 1
        print("epoch : {}, cost change{}".format(iteration, early_stop_criterion))
        cluster = np.eye(num)[cluster]
        return cluster, centroids

    def plot_k_means(self, X, responsibility, centroids):
        random_colors = np.random.random((self.K, 3))
        colors = responsibility.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors, s=20)
        for center in centroids:
            plt.scatter(center[0], center[1], c="black", s=100, marker='x', edgecolors='black')
        plt.axis('equal')
        plt.show()

    def image_cluster(self, image_array, label, type='hard_kmeans'):
        W, H, C = image_array.shape
        reshape_img = image_array.reshape(-1,3).copy()
        clusters, centroids = self.fit(reshape_img)
        centroids = centroids * 255
        cluster_img = (clusters.dot(centroids)).astype('uint8')
        kmeans_image = Image.fromarray(cluster_img.reshape(W, H, C))
        kmeans_image.save('./results/{}/{}.jpg'.format(type, label))
        return reshape_img.reshape(W, H, C)

    def distortion_measure_plot(self, label, type='hard_kmeans'):
        plt.figure()
        plt.plot(self.history)
        plt.xlabel("Iteration")
        plt.ylabel("Distortion measure")
        plt.title("{}".format(label))
        plt.savefig("./results/{}/graph/{}.jpg".format(type,label))
        # plt.show()

# def assign_step2(self, X, centroids):
#     N, D = X.shape
#     euclidean_dist = []
#     for point in X:
#         dist_list = []
#         for center in centroids:
#             dist_list.append(np.linalg.norm(center - point))
#         euclidean_dist.append(dist_list)
#     cluster = np.argmin(np.array(euclidean_dist), axis=1)
#     return cluster, euclidean_dist

# def make_plot(self, X, cluster, centroids):
#     cluster = np.argmin(cluster, axis=1)
#     c_lst = [plt.cm.gist_rainbow(a) for a in np.linspace(0.0, 1.0, len(np.unique(cluster)))]
#     markers = ["o", "^", "v", "<", ">", "s", "+", "h", "D", "1"]
#     pred_X = np.append(X, cluster.reshape(-1,1), axis=1)
#     for c in np.unique(cluster):
#         marker_idx = c % 10
#         plt.scatter(pred_X[pred_X[: , -1] == c, 0], pred_X[pred_X[: , -1] == c, 1], marker=markers[c], s=40, color=c_lst[c], label='class{}'.format(c))
#         plt.scatter(centroids[c][0], centroids[c][1], marker='x', s=100, color='black', edgecolors='black')
#     plt.show()

# plt.scatter(pred_X[pred_X[: , -1] == 1, 0], pred_X[pred_X[: , 2] == 1, 1], marker='^', s=40, color='tab:red', edgecolors='black', label='class2')
# plt.scatter(centroids[1][0], centroids[1][1], marker='x', s=100, color='tab:orange', edgecolors='white')

# for c in np.unique(cluster):
#     reshape_img[np.where(cluster==c)] = centroids[c].reshape(-1,3)
# reshape_img = reshape_img.astype('uint8')
