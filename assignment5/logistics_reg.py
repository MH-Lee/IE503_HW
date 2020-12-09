import numpy as np
import pandas as pd
import random
from numpy.linalg import pinv, inv
from itertools import combinations
'''
logsistic regression using Newton-Raphson method
Author : MH-Lee
Date   : 2020-12-09
'''
def create_10_fold(df):
    n = 10
    sample_idx = list(range(df.shape[0]))
    random.shuffle(sample_idx)
    sample_list = np.array([sample_idx[i * n:(i + 1) * n] for i in range((len(sample_idx) + n - 1) // n )] )
    for tr_idx in combinations(range(10), 9):
        te_idx = tuple(set(range(10)).difference(tr_idx))
        train_sample = sample_list[tr_idx,:].flatten()
        test_sample = sample_list[te_idx,:].flatten()
        np.random.shuffle(train_sample)
        yield (train_sample , test_sample)

def make_fold_report(X, y, verbose=False):
    fold_data = pd.DataFrame(columns=['Accuracy', 'recall', 'precision', 'specificity', 'f1-score'])
    k_fold = create_10_fold(X)
    fold_cm_mat = list()
    auc_list = list()
    for i, (tr, te) in enumerate(k_fold):
        lr_fold = LogisticRegression(max_iter=1000, verbose=verbose)
        lr_fold.fit(X[tr], y[tr])
        y_pred = lr_fold.predict(X[te])
        cf_mat = lr_fold.confusion_mat(y[te], y_pred)
        fold_cm_mat.append(cf_mat)
        if verbose:
            print("================ Fold {} ================".format(str(i+1).zfill(2)))
            lr_fold.cls_metric()
            print("=========================================")
            print()
        else:
            lr_fold.cls_metric(verbose=verbose)
        fold_data = fold_data.append({'Accuracy': lr_fold.acc, 'recall': lr_fold.recall, \
                                      'precision': lr_fold.precision, 'specificity': lr_fold.tnr,
                                      'f1-score': lr_fold.f1_score}, ignore_index=True)
        auc = lr_fold.roc_auc_score(X[te], y[te])
        auc_list.append(auc)
    fold_data['AUC'] = auc_list
    fold_data = fold_data.append(fold_data.mean(), ignore_index=True)
    fold_data = fold_data.append(fold_data.std(), ignore_index=True)
    fold_data.index = ['CV {}'.format(str(i+1).zfill(2)) for i in range(10)] + ['mean', 'std']
    return fold_data, fold_cm_mat


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
                if self.verbose:
                    print("Convergence iter: {}, loss : {} , delta_nll : {}".format(i, nll, abs(delta_nll)))
            if self.convergence:
                break

    def predict_prob(self,X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pro = self.sigmoid(X, self.w)
        return y_pro.flatten()

    def predict(self, X, threshold=0.5):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pro = self.sigmoid(X, self.w)
        y_pre = [1 if x > threshold else 0 for x in y_pro]
        return np.array(y_pre)

    def confusion_mat(self, y_true, y_pred, outputs=True):
        y_pred = pd.Categorical(y_pred, categories=[0,1])
        y_true = pd.Categorical(y_true, categories=[0,1])
        cm_mat = pd.crosstab(y_true, y_pred, \
                             rownames=['Actual'], \
                             colnames=['Predicted'],\
                             dropna=False)
        self.cm_mat = cm_mat
        if  outputs:
            return self.cm_mat

    def cls_metric(self, verbose=True):
        tn, fp, fn, tp = self.cm_mat.values.flatten()
        self.acc = (tp + tn) / self.cm_mat.values.sum()
        self.recall = tp / (tp + fn)
        self.precision = tp / (tp + fp)
        self.tnr = tn / (tn + fp)
        self.f1_score = 2 * ((self.recall * self.precision) / (self.recall + self.precision))
        if verbose:
            print("accuracy : ", round(self.acc, 4))
            print("recall(sensitivity) : ", round(self.recall, 4))
            print("precision : ", round(self.precision, 4))
            print("True Negative Rate(specificity) : ", round(self.tnr, 4))
            print("f1-score : ", round(self.f1_score, 4))

    def roc_auc_score(self, X, y):
        thresholds = np.linspace(0, 1, 20)
        roc_points = []
        tpr_list = []
        fpr_list = []
        for threshold in thresholds:
            y_pred = self.predict(X, threshold=threshold)
            self.confusion_mat(y, y_pred, outputs=False)
            self.cls_metric(verbose=False)
            tpr_list.append(self.recall)
            fpr_list.append(1 - self.tnr)
        auc = 1 + np.trapz(fpr_list,tpr_list)
        # print('Area under curve={}'.format(auc))
        return auc
