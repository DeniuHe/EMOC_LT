"""
Batch Mode Active Ordinal Classification With Expected Model Output Change
"""
import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from numpy.linalg import inv
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class Batch_EMOC_DPC():
    def __init__(self,X, y, labeled, budget, batch, X_test, y_test):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.M = np.array([[(i - j) ** 2 for i in range(self.nClass)] for j in range(self.nClass)])
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.distMatrix = pairwise_distances(X=self.X, metric='euclidean')
        self.K = -self.distMatrix
        self.c = 0.01  #1/C C = 100

        self.model_initialization()
        self.budgetLeft = deepcopy(budget)
        self.batch = batch
        # -------------------------------
        self.MZElist = []
        self.MAElist = []
        self.F1list = []
        self.ALC_MZE = 0.0
        self.ALC_MAE = 0.0
        self.ALC_F1 = 0.0
        self.Redundancy = 0.0
        # -------------------------------

    def model_initialization(self):
        # ------------训练初始KELMOC模型----------------
        n = len(self.labeled)
        self.T_labeled = self.M[self.y[self.labeled],:]
        self.Kernel_labeled = self.K[np.ix_(self.labeled,self.labeled)]
        self.Kernel_labeled_inv = np.linalg.inv(self.c * np.eye(n) + self.Kernel_labeled)
        self.Beta = self.Kernel_labeled_inv @ self.T_labeled

        # ------------生成初始无标记样本池----------------
        self.unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            self.unlabeled.remove(idx)
        return self

    def Batch_Block_Matrix_Inverse(self,A11_inv, A12, A21, A22):
        """
        A11 A12  -1    ==   B11   B12
        A21 A22             B21  B22
        """
        n = A11_inv.shape[0]
        m = A22.shape[0]
        new_M = np.zeros((m+n, m+n))
        B22 = inv(A22 - A21 @ A11_inv @ A12)
        new_M[n:,n:] = B22
        new_M[:n,:n] = A11_inv + (A11_inv @ A12) @ B22 @ (A21 @ A11_inv)
        new_M[:n,n:] = -A11_inv @ A12 @ B22
        new_M[n:,:n] = -B22 @ A21 @ A11_inv
        return new_M


    def Point_Block_Matrix_Inverse(self,A11_inv, A12, A21, A22):
        n = A11_inv.shape[0]
        M = np.zeros((n+1, n+1))
        B22 = inv(A22 - A21 @ A11_inv @ A12)
        M[n,n] = B22
        M[:n,:n] = A11_inv + (A11_inv @ A12) @ B22 @ (A21 @ A11_inv)
        M[:n,n] = (-A11_inv @ A12 @ B22).reshape(-1)
        M[n,:n] = (-B22 @ A21 @ A11_inv).reshape(-1)
        return M



    def model_incremental_train(self, new_ids):
        A11_inv = self.Kernel_labeled_inv
        A12 = self.K[np.ix_(self.labeled, new_ids)]
        A21 = A12.T
        A22 = self.K[np.ix_(new_ids, new_ids)] + self.c * np.eye(len(new_ids))
        Kernel_Bar_Inv = self.Batch_Block_Matrix_Inverse(A11_inv=A11_inv, A12=A12, A21=A21, A22=A22)
        T_Bar = np.vstack((self.T_labeled, self.M[self.y[new_ids],:]))  # 已经验证该行无误
        Beta_Bar = Kernel_Bar_Inv @ T_Bar
        # -------------------------
        self.Kernel_labeled_inv = Kernel_Bar_Inv
        self.T_labeled = T_Bar
        self.Beta = Beta_Bar



    def tmp_incremental_train(self, tmp_idx):
        A11_inv = self.Kernel_labeled_inv
        A12 = self.K[np.ix_(self.labeled, [tmp_idx])]
        A21 = A12.T
        A22 = self.K[tmp_idx, tmp_idx] + self.c
        K_bar_inv = self.Point_Block_Matrix_Inverse(A11_inv=A11_inv, A12=A12, A21=A21, A22=A22)
        return K_bar_inv


    def predict_proba(self, X):
        K = -pairwise_distances(X,self.X[self.labeled], metric='euclidean')
        coded_tmp = K.dot(self.Beta)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix


    def predict(self, X):
        K = -pairwise_distances(X=X, Y=self.X[self.labeled], metric="euclidean")
        # print("K:",K.shape)
        # print("Beta:",self.Beta.shape)
        coded_preds = K.dot(self.Beta)
        predictions = np.argmin(np.linalg.norm(coded_preds[:, None] - self.M, axis=2, ord=1), axis=1)
        # predictions = self.le_.inverse_transform(predictions)
        return predictions


    def evaluation(self):
        y_hat = self.predict(X=self.X_test)
        self.MZElist.append(1-accuracy_score(self.y_test, y_hat))
        self.MAElist.append(mean_absolute_error(self.y_test, y_hat))
        self.F1list.append(f1_score(self.y_test, y_hat, average='macro'))


    def select(self):
        self.evaluation()
        while self.budgetLeft > 0:
            score = np.zeros(len(self.unlabeled))
            proba_matrix = self.predict_proba(X=self.X[self.unlabeled])
            Beta_0 = np.vstack((self.Beta, np.zeros(self.nClass)))
            for i, idx in enumerate(self.unlabeled):
                score[i] = 0.0
                K_bar_inv = self.tmp_incremental_train(tmp_idx=idx)
                tmp_labeled = deepcopy(self.labeled)
                tmp_labeled.append(idx)
                for j in range(self.nClass):
                    T_bar = np.vstack((self.T_labeled,self.M[j]))
                    Beta_bar = K_bar_inv @ T_bar
                    delta_Beta = Beta_0 - Beta_bar
                    output = self.K[np.ix_(range(self.nSample), tmp_labeled)].dot(delta_Beta)
                    output_score = np.sum(abs(output))
                    score[i] += proba_matrix[i,j] * output_score

            # =============过滤60%的样本（emoc分数小的样本）=============
            candidate = []
            candidate_score = []
            order_score = np.flipud(np.argsort(score))   #根据emoc分数降序排序
            candi_size = int(len(self.unlabeled)*0.5)    #设定保留比例40%的
            if candi_size < self.batch:  # 防止剔除后剩余样本数目不足batch
                self.batch = candi_size
            for i in range(candi_size):  # 保存保留样本的索引和emoc分数
                candidate.append(self.unlabeled[order_score[i]])
                candidate_score.append(score[order_score[i]])

            # ==========执行密度峰值聚类算法=============
            rho = candidate_score  # 使用emoc分数作为密度
            leader = np.ones(len(candidate),dtype=int) * int(-1) #领导是比自己密度大且距离自己最近样本
            delta = np.zeros(len(candidate),dtype=float)  #初始化delta，到领导的距离。
            order_rho = np.flipud(np.argsort(rho))  #对密度排序
            # ---------------计算密度最大点的delta--------------------
            delta[order_rho[0]] = np.max(self.distMatrix[candidate[order_rho[0]],:])  # 对密度最大点赋予一个较大的领导距离
            # print("max_delta:",np.max(self.distMatrix[candidate[order_rho[0]],:]))
            # ---------------计算非密度最大点的delta------------------
            for i in range(1, len(candidate)):
                min_dist = np.inf
                min_idx = -1
                for j in range(i):
                    dist = self.distMatrix[candidate[order_rho[i]], candidate[order_rho[j]]]
                    # candidate[order_rho[i] 密度排序为第i大的样本的索引，
                    # candidate[order_rho[j]] 密度排序为第j大的样本的索引。
                    if dist < min_dist:   # 找与i最近的密度比i大的样本
                        min_dist = dist
                        min_idx = order_rho[j]
                delta[order_rho[i]] = min_dist
                leader[order_rho[i]] = min_idx   # leader里面不是candidate的真实索引，是排序索引
            # print("max_delta:",delta)
            # ------------计算得到gamma指标----------------
            gamma = rho * delta
            order_gamma = np.flipud(np.argsort(gamma))

            selected_ids = []
            for i in range(self.batch):
                selected_ids.append(candidate[order_gamma[i]])





            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



            n1 = np.linspace(0,5,1000)
            m1 = np.sqrt(25 - n1**2)

            n2 = np.linspace(0,9,1000)
            m2 = np.sqrt(81 - n2**2)

            plt.style.context('seaborn-while')
            plt.plot(n1,m1,ls='--',lw=1,color='k')
            plt.plot(n2,m2,ls='--',lw=1,color='k')


            ids_0 = []
            jds_0 = []



            for i, idx in enumerate(self.unlabeled):
                if y[idx] == 0:
                    ids_0.append(idx)
                    jds_0.append(i)
            # plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=np.asarray(rho)[jds_0],marker='o',cmap='viridis')
            plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=score[jds_0],marker='o',cmap='viridis')

            ids_0 = []
            jds_0 = []
            # for i, idx in enumerate(candidate):
            for i, idx in enumerate(self.unlabeled):
                if y[idx] == 1:
                    ids_0.append(idx)
                    jds_0.append(i)
            # plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=np.asarray(rho)[jds_0],marker='v',cmap='viridis')
            plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=score[jds_0],marker='v',cmap='viridis')

            ids_0 = []
            jds_0 = []
            # for i, idx in enumerate(candidate):
            for i, idx in enumerate(self.unlabeled):
                if y[idx] == 2:
                    ids_0.append(idx)
                    jds_0.append(i)
            # plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=np.asarray(rho)[jds_0],marker=',',cmap='viridis')
            plt.scatter(self.X[ids_0][:,0],self.X[ids_0][:,1], c=score[jds_0],marker=',',cmap='viridis')

            for i, idx in enumerate(candidate):
                jdx = candidate[leader[i]]
                if leader[i]!= -1:
                    plt.plot(self.X[[idx,jdx]][:,0], self.X[[idx,jdx]][:,1],c='k',lw=1.5)





            plt.scatter(self.X[selected_ids][:,0], self.X[selected_ids][:,1],c='w',edgecolors='r',marker="o",linewidths=3,s=120)
            plt.scatter(self.X[self.labeled][:,0], self.X[self.labeled][:,1],c='r',marker="*",s=150)
            plt.colorbar()
            # plt.title("already label {} instances".format(len(self.labeled)-3))
            plt.tight_layout()
            plt.show()
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            print("selected::",selected_ids)

            self.model_incremental_train(new_ids=selected_ids)
            # ----------将选择的样本从无标记样本池中剔除--------------
            for idx in selected_ids:
                self.unlabeled.remove(idx)
            # ----------将选择的样本加入训练样本集-------------------
            for idx in selected_ids:
                self.labeled.append(idx)

            self.budgetLeft -= self.batch
            self.evaluation()

        for i in range(len(self.MZElist)-1):
            self.ALC_MZE += 0.5 * (self.MZElist[i] - self.MZElist[i+1]) * self.batch
            self.ALC_MAE += 0.5 * (self.MAElist[i] - self.MAElist[i+1]) * self.batch
            self.ALC_F1 += 0.5 * (self.F1list[i] - self.F1list[i+1]) * self.batch

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X=self.X[self.labeled])
        self.Redundancy = np.mean(neigh.kneighbors()[0].flatten())
        # print(neigh.kneighbors()[0].flatten())






if __name__ == '__main__':
    data_path = Path(r"D:\OCdata")
    name = "example2"
    read_data_path = data_path.joinpath(name + ".csv")
    data = np.array(pd.read_csv(read_data_path, header=None))
    X = np.asarray(data[:, :-1], np.float64)
    y = data[:, -1].astype(int)
    nClass = len(np.unique(y))
    Budget = 20 * nClass
    # labeled = [504,101,104]
    labeled = [504, 101, 104, 100, 250, 749, 339, 316]
    batch = 4

    model = Batch_EMOC_DPC(X=X, y=y, labeled=labeled, budget=Budget, batch=batch, X_test=X, y_test=y)
    model.select()
    print("冗余度：",model.Redundancy)