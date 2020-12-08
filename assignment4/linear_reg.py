import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import probplot

class LinearReg:
    def __init__(self, penalty=None, alpha=None,\
                 max_iter=None, fit_intercept=True,
                 learning_rate=1e-3, verbose=True):
        '''
        penalty : l1 or l2
        alpha  : Strength of regularization
        '''
        if penalty is not None:
            if alpha is None:
                self.alpha = 1.0
            else:
                self.alpha = alpha
            self.mode = "lasso" if penalty == 'l1' else 'ridge'

            if self.mode == 'lasso':
                if max_iter is None:
                    self.max_iter = 100
                else:
                    self.max_iter = max_iter
                self.learning_rate = learning_rate
                if verbose:
                    print("iter : {}".format(self.max_iter))
            if verbose:
                print("alpha set {}".format(self.alpha))
        else:
            self.mode = 'vanilla'
        self.w = None
        self.fit_intercept = fit_intercept
        if verbose:
            print("{} regression is start !".format(self.mode))

    def r2_score(self, y_true, y_pred):
        '''
        SSR = residual sum of square
        SST = total sum of square
        R2 = 1 - SSR/TSS
        '''
        SSR = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        SST = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
        nonzero_denominator = SSR != 0
        nonzero_numerator = SST != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([y_true.shape[1]])
        output_scores[valid_score] = 1 - (SSR[valid_score] / SST[valid_score])
        return np.average(output_scores)

    def softThreshold(self, rho, lambdas):
        if (rho < -lambdas):
            return rho + lambdas
        elif (rho > lambdas):
            return rho - lambdas
        else:
            return 0.

    def fit(self, X, y, disable=False):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
            # X = np.append(np.ones(X.shape[0]).reshape(-1, 1), X, axis=1)
        if self.mode == 'vanilla':
            try:
                self.w = np.linalg.inv(X.T @ X, X.T @ y)
            except:
                self.w = np.linalg.pinv(X) @ y
            self.cal_residual(X, y)
        elif self.mode == 'ridge':
            gram = X.T @ X
            regularizer = self.alpha * np.eye(np.size(X, 1))
            self.w = np.linalg.solve(regularizer + gram,  X.T @ y)
            self.cal_residual(X, y)
        else:
            # raise Exception("not implemented")
            N, D = X.shape
            if y.ndim == 1:
                y = y.reshape(-1,1)
                _, n_target = y.shape
                W = np.zeros((D, n_target))
            else:
                _, n_target = y.shape
                W = np.zeros((D, n_target))

            if self.fit_intercept:
                W[0] = np.sum(y - np.dot(X[:, 1:], W[1:]) ,axis=0)/(X.shape[0])

            for i in tqdm(range(self.max_iter), disable=disable):
                start = 1 if self.fit_intercept else 0
                for j in range(start, len(W)):
                    W_tmp = W.copy()
                    W_tmp[j] = [0.0] * n_target
                    r_j = y - np.dot(X, W_tmp)
                    rho = np.dot(X[:, j], r_j)
                    lambdas = self.alpha * X.shape[0]

                    for k in range(len(rho)):
                        W[j, k] = self.softThreshold(rho[k], lambdas) / (X[:, j]**2).sum()

                    if self.fit_intercept:
                        W[0] = np.sum(y - np.dot(X[:, 1:], W[1:]) ,axis=0)/(X.shape[0])
            self.w = W
            self.cal_residual(X, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y = X.dot(self.w)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        r2_score = self.r2_score(y, y_pred)
        return r2_score

    def cal_residual(self, X, y):
        y_hat = X.dot(self.w)
        self.residual = y - y_hat

    def rmse(self, y_true, y_pred):
        return np.sqrt(((y_pred - y_true) ** 2).mean())
    '''
    this is only used make a Assingment4 report
    '''
    def report_HW(self, train_X, test_X, train_y, test_y, report_type='r2', make_df=False):
        y_tr_pred = self.predict(train_X)
        y_te_pred = self.predict(test_X)
        self.total_tr_r2 = self.score(train_X, train_y)
        self.total_te_r2 = self.score(test_X, test_y)
        self.lat_tr_r2 = self.r2_score(train_y[:,0].reshape(-1,1), y_tr_pred[:,0].reshape(-1,1))
        self.lat_te_r2 = self.r2_score(test_y[:,0].reshape(-1,1), y_te_pred[:,0].reshape(-1,1))
        self.long_tr_r2 = self.r2_score(train_y[:,1].reshape(-1,1), y_tr_pred[:,1].reshape(-1,1))
        self.long_te_r2 = self.r2_score(test_y[:,1].reshape(-1,1), y_te_pred[:,1].reshape(-1,1))
        self.total_tr_rmse = self.rmse(train_y, y_tr_pred)
        self.total_te_rmse = self.rmse(test_y, y_te_pred)
        self.lat_tr_rmse = self.rmse(train_y[:,0].reshape(-1,1), y_tr_pred[:,0].reshape(-1,1))
        self.lat_te_rmse = self.rmse(test_y[:,0].reshape(-1,1), y_te_pred[:,0].reshape(-1,1))
        self.long_tr_rmse = self.rmse(train_y[:,1].reshape(-1,1), y_tr_pred[:,1].reshape(-1,1))
        self.long_te_rmse = self.rmse(test_y[:,1].reshape(-1,1), y_te_pred[:,1].reshape(-1,1))
        if report_type == 'r2':
            print("Train total R-square :", round(self.total_tr_r2, 4))
            print("Test  total R-square :", round(self.total_te_r2, 4))
            print("Train latitude R-square :", round(self.lat_tr_r2, 4))
            print("Test  latitude R-square :", round(self.lat_te_r2, 4))
            print("Train longitude R-square :", round(self.long_tr_r2, 4))
            print("Test  longitude R-square :", round(self.long_te_r2, 4))
        else:
            print("Train total RMSE :", round(self.total_tr_rmse, 4))
            print("Test  total RMSE :", round(self.total_te_rmse, 4))
            print("Train latitude RMSE :", round(self.lat_tr_rmse, 4))
            print("Test  latitude RMSE :", round(self.lat_te_rmse, 4))
            print("Train longitude RMSE :", round(self.long_tr_rmse, 4))
            print("Test  longitude RMSE :", round(self.long_te_rmse, 4))


def resid_plot(model, model_name='linear reg'):
    std_resid = (model.residual - model.residual.mean(axis=0))/ (model.residual.std(axis=0))
    fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(10,4))
    st = fig.suptitle("Residual plot {}".format(model_name), fontsize="x-large")
    for i in range(len(axes)):
        y_label = ['latitude', 'longitude']
        axes[i].scatter(range(len(std_resid)), std_resid[:,i])
        axes[i].axhline(y=0, **{'ls':'--', 'c':'tab:red'})
        axes[i].set_xlabel("{}_fitted".format(y_label[i]))
        axes[i].set_ylabel("Standardized resdidual")
    plt.show()

def plot_real_vs_predicted(y_true, y_pred, ax=None, mode=None):
    normalized_yt = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    normalized_yp = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    ax.plot(normalized_yp, normalized_yt,'.', color='tab:blue')
    ax.plot([-0.02, 1.1],[-0.02, 1.1], 'g-', color='tab:red')
    ax.set_xlim(-0.02, 1.1)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel('Predicted ({})'.format(mode))
    ax.set_ylabel('True ({})'.format(mode))

def normality_plot(model):
    f, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(axes.shape[0]):
        y_label = ['latitude', 'longitude']
        axes[0][i].boxplot(model.residual[:,i])
        probplot(model.residual[:,i], plot=axes[1][i])
        axes[0][i].set_title("{} boxplot and qqplot".format(y_label[i]))
    plt.show()

def grid_search_df(train_X, train_y, test_X, test_y, \
                   lambdas_list=None, penalty='l1', disable=True):
    train_r2_score, test_r2_score = list(), list()
    train_rmse, test_rmse = list(), list()
    lambdas_list = lambdas_list
    for lambda_ in tqdm(lambdas_list):
        model = LinearReg(penalty=penalty, alpha=lambda_, verbose=False)
        model.fit(train_X, train_y, disable=disable)
        y_tr_pred = model.predict(train_X)
        y_te_pred = model.predict(test_X)
        tr_r2_ = model.score(train_X, train_y)
        te_r2_ = model.score(test_X, test_y)
        tr_rmse = model.rmse(train_y, y_tr_pred)
        te_rmse = model.rmse(test_y, y_te_pred)
        train_r2_score.append(tr_r2_)
        test_r2_score.append(te_r2_)
        train_rmse.append(tr_rmse)
        test_rmse.append(te_rmse)

    score_df = pd.DataFrame([lambdas_list, train_r2_score, test_r2_score, train_rmse, test_rmse],
                            index=['lambda','tr_r2', 'te_r2', 'tr_rmse', 'te_rmse']).T
    score_df.set_index('lambda', inplace=True)
    return score_df

def plot_r2_rmse(df, lambdas_list):
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    df[['tr_r2', 'te_r2']].plot(ax=axes[0])
    df[['tr_rmse', 'te_rmse']].plot(ax=axes[1])
    axes[0].set_title('R-square')
    axes[1].set_title('RMSE')
    axes[0].set_ylabel('R-square')
    axes[1].set_ylabel('RMSE')
    r2_armax = np.argmax(df.values[:,0:2].round(3), axis=0).tolist()
    rmse_argmin = np.argmin(df.values[:,2:].round(3), axis=0).tolist()
    axes[0].axvline(lambdas_list[r2_armax[0]], c='lime', ls='--', label="argmax train r2")
    axes[0].axvline(lambdas_list[r2_armax[1]], c='tab:red', ls='--', label="argmax test r2")
    axes[1].axvline(lambdas_list[rmse_argmin[0]], c='lime', ls='--', label="argmin train rmse")
    axes[1].axvline(lambdas_list[rmse_argmin[1]], c='tab:red', ls='--', label="argmin test rmse")
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
