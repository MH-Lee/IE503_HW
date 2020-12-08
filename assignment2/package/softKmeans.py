import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from package.kmeans import Kmeans
from utils.make_dataset import make_dataset
np.random.seed(100)

class SoftKmeans(Kmeans):
    def __init__(self, K=2, beta=10.0, early_stop=10, max_iter=500,\
                 initial_mode='random', opt_method='cost_fuction'):
        super().__init__(initial_mode=initial_mode)
        self.K = K
        self.beta = beta
        self.early_stop = early_stop
        self.opt_method = opt_method
        self.max_iter = max_iter
        if self.K > 20:
            self.epsilon = 1e-3 * 4
        else:
            self.epsilon = 1e-3
        # self.history = []
        print("initialize mode : {}, optimize method : {}, max_iter : {}".format(self.initial_mode, self.opt_method, self.max_iter))

    def assign_step(self, X, centroids):
        N, _ = X.shape
        K, D = centroids.shape
        responsibility = np.zeros((N, self.K))
        for k in range(K):
            responsibility[:, k] = np.exp(-self.beta * np.linalg.norm(X - centroids[k],axis=1))
        responsibility /= responsibility.sum(axis=1, keepdims=True)
        return responsibility

    def update_step(self, X, responsibility):
        N, D = X.shape
        centers = np.zeros((self.K, D))
        for k in range(self.K):
            centers[k] = responsibility[:, k].dot(X) / responsibility[:, k].sum()
        return centers

    def object_func(self, X, responsibility, centroids):
        energy = 0
        entropy = 0
        for k in range(self.K):
            x_mean_norm = np.linalg.norm(X - centroids[k], axis=1)
            energy += x_mean_norm.dot(responsibility[:,k])
            entropy += responsibility[:,k].dot(-np.log(responsibility[:,k]))
        cost = energy - (1.0/self.beta) * entropy
        return energy

    def fit(self, X):
        if self.initial_mode == 'image':
            centroids = self.initialize_centroid()/255
        else:
            centroids = self.initialize_centroid(X)
        print("initial_mode : ", self.initial_mode, "intiial centroids : ", centroids)
        old_dm = float('inf') if self.opt_method == 'cost_fuction' else centroids
        early_stop_list = [0]
        iteration = 0
        self.history.append(old_dm)
        while sum(early_stop_list) < self.early_stop:
            if iteration > self.max_iter:
                break
            responsibility = self.assign_step(X, centroids)
            centroids = self.update_step(X, responsibility)
            ### set early_stop
            if self.opt_method == 'cost_fuction':
                new_dm = self.object_func(X, responsibility, centroids)
                early_stop_criterion = old_dm - new_dm
                if early_stop_criterion < 0:
                    break
            elif self.opt_method == 'distance':
                new_dm = centroids
                early_stop_criterion = np.sqrt(np.linalg.norm(old_dm - new_dm, axis=1).mean())
            else:
                raise ValueError(f"Select cost_fuction or distance")
            if iteration % 20 == 0:
                # print(responsibility)
                print("Early stop count : {}/{}".format(sum(early_stop_list), self.early_stop))
                if self.opt_method == 'cost_fuction':
                    print("epoch : {}, cost change : {}".format(iteration, early_stop_criterion))
                else:
                    print("epoch : {}, distance change : {}".format(iteration, early_stop_criterion))
            if abs(early_stop_criterion) <= self.epsilon:
                early_stop_list.append(1)
            else:
                early_stop_list.append(0)
            if abs(early_stop_criterion) <= 1e-3:
                break
            old_dm = new_dm
            self.history.append(old_dm)
            iteration += 1
        print("epoch : {}, change value(cost or distance) : {}".format(iteration, early_stop_criterion))
        return responsibility, centroids

    # def plot_k_means(self, X, responsibility, centroids):
    #     random_colors = np.random.random((self.K, 3))
    #     colors = responsibility.dot(random_colors)
    #     plt.scatter(X[:,0], X[:,1], c=colors)
    #     for center in centroids:
    #         plt.scatter(center[0], center[1], c="black", s=100, marker='x', edgecolors='black')
    #     plt.show()
