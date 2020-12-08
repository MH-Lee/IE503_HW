import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal


class MFA:
    def __init__(self, X, n_cluster=None, n_factor=None):
        # mean_x = np.mean(X, axis=1, keepdims=True)
        self.X = X
        self.D, self.N = X.shape
        # set hyper parameter
        '''
        n_factor : The number of latent value (q)
        n_cluster : The number of cluster
        '''
        self.n_factor = n_factor
        self.n_cluster = n_cluster
        # E_step parameter
        self.Es = None
        self.Ess = None
        self.Rjt = None
        # M_step parameter
        self.pi = None
        self.mu = None
        self.A = None
        self.sigma = None

    def _initializer(self):
        self.mu = self.X.copy()
        np.random.shuffle(self.mu)
        self.mu = self.mu[:,:self.n_cluster].T.reshape(self.n_cluster, self.D, 1)
        self.A = np.ones((self.n_cluster, self.D, self.n_factor))
        self.pi = np.array([1] * self.n_cluster) / self.n_cluster  # (K,) array
#         self.sigma = np.diag(np.arange(self.D) + 1 + np.exp(-700))
        self.sigma = np.diag(np.diag(self.X.dot(self.X.T)))
        self.cll = [np.inf]

    def E_step(self):
        Es = np.zeros((self.n_cluster, self.N, self.n_factor)) 
        Ess = np.zeros((self.n_cluster, self.N, self.n_factor, self.n_factor))
        Rjt = np.zeros((self.n_cluster, self.N))
        X_center = self.X - self.mu
        for j in range(self.n_cluster):
            Rjt[j] = self.pi[j] * multivariate_normal.pdf(self.X.T, self.mu[j].flatten(), self.A[j,:]@self.A[j,:].T + self.sigma)
            phi = self.A[j,:].T @ np.linalg.inv(self.A[j,:] @ self.A[j,:].T + self.sigma) # phi
        
            Es[j] = (phi @ X_center[j]).T 
            for t in range(self.N):
                Ess[j,t] = np.eye(self.n_factor) - phi @ self.A[j] + phi @ X_center[j,:, t].reshape(-1,1) @ X_center[j,:, t].reshape(-1,1).T @ phi.T
        Rjt = Rjt/Rjt.sum(axis=0, keepdims=True)
        self.Es, self.Ess, self.Rjt =  Es, Ess, Rjt

    def M_step(self):
        # new A
        A_new = np.zeros(shape=(self.n_cluster, self.D, self.n_factor))
        for j in range(self.n_cluster):
            Aj1 = np.zeros(shape=(self.D, self.n_factor))
            Aj2 = np.zeros(shape=(self.n_factor, self.n_factor))
            for t in range(self.N):
                Aj1 += self.Rjt[j,t] * ((self.X[:,t].reshape(-1,1) - self.mu[j]) @ self.Es[j,t].reshape(1,-1))
                Aj2 += self.Rjt[j,t] * self.Ess[j,t]
            A_new[j,:,:] = Aj1 @ np.linalg.inv(Aj2)
        # new mu
        mu_new = np.zeros(shape=(self.D, self.n_cluster))
        for j in range(self.n_cluster):
            temp_muj = np.zeros(shape=(self.D, 1))
            for t in range(self.N):
                temp_muj += self.Rjt[j,t]*((self.X[:,t].reshape(-1,1) - A_new[j,:,:] @ self.Es[j,t].reshape(-1,1)))
            mu_new[:,j] = (temp_muj/self.Rjt[j,:].sum()).flatten()
        mu_new = mu_new.T[:,:,np.newaxis]
        # new sigma
        sigma_new = np.zeros(shape=(self.D, self.D))
        for j in range(self.n_cluster):
            for t in range(self.N):
                sigma_new += self.Rjt[j,t]*((self.X[:,t].reshape(-1,1) - mu_new[j]) @ (self.X[:,t].reshape(-1,1) - mu_new[j]).T - (A_new[j,:,:] @ self.Es[j,t].reshape(-1,1) @ (self.X[:,t].reshape(-1,1) - mu_new[j]).T))
        sigma_new = (1/self.N)*np.diag(np.diag(sigma_new))
        # new pi
        pi_new = self.Rjt.mean(axis=1)
        self.A, self.mu, self.sigma, self.pi =  A_new, mu_new, sigma_new, pi_new

    def loss_function(self):
        self.loss = 0
        inv_sigma = np.linalg.inv(self.sigma)
        for j in range(self.n_cluster):
            for t in range(self.N):
                loss_temp = np.log(self.pi[j]) - 0.5*(self.X[:,t] @ inv_sigma @ self.X[:,t] - 2 * self.X[:,t] @ inv_sigma @ self.A[j] @ self.Es[j, t] \
                                              -2 * self.X[:,t] @ inv_sigma @ self.mu[j].reshape(-1) + 2 * self.mu[j].reshape(-1) @ inv_sigma @ self.A[j]@ self.Es[j,t]\
                                              + np.matrix.trace(self.A[j].T @ inv_sigma @ self.A[j] @ self.Ess[j,t]) + self.mu[j].reshape(-1) @ inv_sigma @self.mu[j].reshape(-1))
                det_sigma = np.linalg.det(self.sigma + np.exp(-700))
                loss = self.Rjt[j,t]*(loss_temp - 0.5 * np.log(det_sigma) + (self.D/2)*np.log(2*np.pi))
                self.loss += loss
        # print(np.isnan(np.log(det_sigma)))
        print("completed log-likelihood : ", self.loss)

    def fit(self, max_iter=20, tol_el=1e-1):
        iter_ = 0
        self._initializer()
        while iter_ < max_iter:
            self.E_step()
            self.M_step()
            self.loss_function()
            if abs(self.cll[-1] - self.loss) < tol_el:
                print("old_loss : ", self.cll[-1])
                print("new_loss : ", self.loss)
                break
            self.cll.append(self.loss)
            iter_ += 1

    def plot_loglikelihood(self):
        plot_x = [-1*p for p in self.cll]
        plt.plot(plot_x)
        plt.show()