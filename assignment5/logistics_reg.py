import numpy as np
import pandas as pd
from numpy.linalg import pinv, inv
from itertools import combinations
'''
logsistic regression using Newton-Raphson method
Author : MH-Lee
Date   : 2020-12-09
'''
def create_10_fold(sample_list):
    for tr_idx in combinations(range(10), 9):
        te_idx = tuple(set(range(10)).difference(tr_idx))
        train_sample = sample_list[tr_idx,:].flatten()
        test_sample = sample_list[te_idx,:].flatten()
        np.random.shuffle(train_sample)
        yield (train_sample , test_sample)


class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=10000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.neg_log_likelihood = [-np.inf]
        self.convergence = False
        self.w = None

    def sigmoid(self, X, weight):
        z = np.dot(X, weight)
        return 1 / (1 + np.exp(-z) + 1e-6)

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum()

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        # weights initialization
        self.w = np.zeros(X.shape[1])
        for i in range(self.max_iter):
            h = self.sigmoid(X, self.w)
            nll = self.loss(h, y)
            self.neg_log_likelihood.append(nll)
            delta_nll = self.neg_log_likelihood[-2] - self.neg_log_likelihood[-1]
            gradient = np.dot(X.T, (h - y)) / y.size
            R = np.diag(h * (1 - h))
            Hess = X.T @ R @ X
            try:
                self.w -= (inv(Hess) @ gradient.reshape(-1, 1)).flatten()
            except:
                self.w -= (pinv(Hess) @ gradient.reshape(-1, 1)).flatten()
            if np.isnan(nll):
                break
            if (self.verbose and (i % 100 == 0)):
                print("iter: {}, loss : {}".format(i, nll))
            if (i > 1 and not self.convergence and abs(delta_nll) < 1e-5):
                self.convergence = True
                print("Convergence iter: {}, loss : {} , delta_nll : {}".format(i, nll, abs(delta_nll)))
            if self.convergence:
                break

    def predict_prob(self,X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pro = self.sigmoid(X, self.w)
        return y_pro.flatten()

    def predict(self, X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pro = self.sigmoid(X, self.w)
        y_pre = [1 if x > 0.5 else 0 for x in y_pro]
        return np.array(y_pre)

    def confusion_mat(self, y_true, y_pred):
        cm_mat = pd.crosstab(y_true, y_pred, \
                             rownames=['Actual'], \
                             colnames=['Predicted'])
        self.cm_mat = cm_mat.T
        return self.cm_mat

    def cls_metric(self, verbose=True):
        tn, fn, fp, tp = self.cm_mat.values.flatten()
        self.acc = (tp + tn) / self.cm_mat.values.sum()
        self.recall = tp / (tp + fn)
        self.precision = tp / (tp + fp)
        self.tnr = tn / (tn + fp)
        if verbose:
            print("accuracy : ", round(self.acc, 4))
            print("recall(sensitivity) : ", round(self.recall, 4))
            print("precision : ", round(self.precision, 4))
            print("True Negative Rate(specificity) : ", round(self.tnr, 4))
