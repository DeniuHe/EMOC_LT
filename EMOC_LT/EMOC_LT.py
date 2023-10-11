"""
Batch Mode Active Ordinal Classification With Expected Model Output Change
执行密度峰值算法时先进行数据过滤（选择的簇中心点，簇中心点是当前簇密度最大点）
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
np.seterr(divide='ignore',invalid='ignore')
from sklearn.neighbors import NearestNeighbors


class Batch_EMOC():
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
        self.K = -self.distMatrix #Perceptron Kernel
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
            if self.batch > self.budgetLeft:
                self.batch = self.budgetLeft
            reduced_set = np.random.choice(range(self.nSample), size=300, replace=False)
            # reduced_set = range(self.nSample)
            CANDIDATE = np.random.choice(self.unlabeled,size=500,replace=False)
            # CANDIDATE = self.unlabeled
            score = np.zeros(len(CANDIDATE))
            proba_matrix = self.predict_proba(X=self.X[CANDIDATE])
            Beta_0 = np.vstack((self.Beta, np.zeros(self.nClass)))
            for i, idx in enumerate(CANDIDATE):
                # for i, idx in enumerate(self.unlabeled):
                score[i] = 0.0
                K_bar_inv = self.tmp_incremental_train(tmp_idx=idx)
                tmp_labeled = deepcopy(self.labeled)
                tmp_labeled.append(idx)
                for j in range(self.nClass):
                    T_bar = np.vstack((self.T_labeled,self.M[j]))
                    Beta_bar = K_bar_inv @ T_bar
                    delta_Beta = Beta_0 - Beta_bar
                    output = self.K[np.ix_(reduced_set, tmp_labeled)].dot(delta_Beta)
                    output_score = np.sum(abs(output))
                    score[i] += proba_matrix[i,j] * output_score

            candidate = []
            candidate_score = []
            order_score = np.flipud(np.argsort(score))
            candi_size = int(len(CANDIDATE)*0.5)
            # candi_size = int(len(self.unlabeled)*0.4)
            for i in range(candi_size):
                candidate.append(CANDIDATE[order_score[i]])
                # candidate.append(self.unlabeled[order_score[i]])
                candidate_score.append(score[order_score[i]])

            rho = candidate_score
            delta = np.zeros(len(candidate))
            order_rho = np.flipud(np.argsort(rho))
            # ---------------计算密度最大点的delta--------------------
            delta[order_rho[0]] = np.max(self.distMatrix[candidate[order_rho[0]],:])
            # ---------------计算非密度最大点的delta------------------
            for i in range(1, len(candidate)):
                min_dist = np.inf
                for j in range(i):
                    dist = self.distMatrix[candidate[order_rho[i]], candidate[order_rho[j]]]
                    if dist < min_dist:
                        min_dist = dist
                delta[order_rho[i]] = min_dist
            gamma = rho * delta
            order_gamma = np.flipud(np.argsort(gamma))

            selected_ids = []
            # print("candidate::",len(candidate))
            for i in range(self.batch):
                selected_ids.append(candidate[order_gamma[i]])
            self.model_incremental_train(new_ids=selected_ids)
            # ----------将选择的样本从无标记样本池中剔除--------------
            for idx in selected_ids:
                self.unlabeled.remove(idx)
            # ----------将选择的样本加入训练样本集-------------------
            for idx in selected_ids:
                self.labeled.append(idx)

            self.budgetLeft -= self.batch
            self.evaluation()

            self.ALC_MZE += 0.5 * (self.MZElist[-2] + self.MZElist[-1]) * self.batch
            self.ALC_MAE += 0.5 * (self.MAElist[-2] + self.MAElist[-1]) * self.batch
            self.ALC_F1 += 0.5 * (self.F1list[-2] + self.F1list[-1]) * self.batch

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X=self.X[self.labeled])
        self.Redundancy = (1/np.mean(neigh.kneighbors()[0].flatten()))




if __name__ == '__main__':
    method = "EMOC_LT"
    names_list = ["Eucalyptus"]

    class results():
        def __init__(self):
            self.MZEList = []
            self.MAEList = []
            self.F1List = []
            self.ALC_MZE = []
            self.ALC_MAE = []
            self.ALC_F1 = []
            self.Redun = []

    class stores():
        def __init__(self):
            self.Redun_mean = []
            self.Redun_std = []
            #-----------------------
            self.MZEList_mean = []
            self.MZEList_std = []
            # -----------------
            self.MAEList_mean = []
            self.MAEList_std = []
            # -----------------
            self.F1List_mean = []
            self.F1List_std = []
            # -----------------
            # -----------------
            self.ALC_MZE_mean = []
            self.ALC_MZE_std = []
            # -----------------
            self.ALC_MAE_mean = []
            self.ALC_MAE_std = []
            # -----------------
            self.ALC_F1_mean = []
            self.ALC_F1_std = []
            # -----------------
            self.ALC_MZE_list = []
            self.ALC_MAE_list = []
            self.ALC_F1_list = []


    for name in names_list:
        print("########################{}".format(name))
        data_path = Path(r"E:\GGGG_BIYE\DataSet")
        partition_path = Path(r"E:\GGGG_BIYE\Partition")
        """--------------read the whole data--------------------"""
        read_data_path = data_path.joinpath(name + ".csv")
        data = np.array(pd.read_csv(read_data_path, header=None))
        X = np.asarray(data[:, :-1], np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[:, -1]
        y -= y.min()
        nClass = len(np.unique(y))
        Budget = 25 * nClass
        batch = 3

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)
        # --------------------------------------
        RESULT = results()
        STORE = stores()
        # --------------------------------------
        workbook = xlwt.Workbook()
        count = 0
        for SN in book_partition.sheet_names():
            S_Time = time()
            train_idx = []
            test_idx = []
            labeled = []
            table_partition = book_partition.sheet_by_name(SN)
            for idx in table_partition.col_values(0):
                if isinstance(idx,float):
                    train_idx.append(int(idx))
            for idx in table_partition.col_values(1):
                if isinstance(idx,float):
                    test_idx.append(int(idx))
            for idx in table_partition.col_values(2):
                if isinstance(idx,float):
                    labeled.append(int(idx))

            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = Batch_EMOC(X=X_train, y=y_train, labeled=labeled, budget=Budget, batch=batch, X_test=X_test, y_test=y_test)
            model.select()
            RESULT.MZEList.append(model.MZElist)
            RESULT.MAEList.append(model.MAElist)
            RESULT.F1List.append(model.F1list)
            RESULT.ALC_MZE.append(model.ALC_MZE)
            RESULT.ALC_MAE.append(model.ALC_MAE)
            RESULT.ALC_F1.append(model.ALC_F1)
            RESULT.Redun.append(model.Redundancy)

            print("SN===",SN, "time:",time()-S_Time)


        STORE.Redun_mean = np.mean(RESULT.Redun)
        STORE.Redun_std = np.std(RESULT.Redun)
        STORE.MZEList_mean = np.mean(RESULT.MZEList, axis=0)
        STORE.MZEList_std = np.std(RESULT.MZEList, axis=0)
        STORE.MAEList_mean = np.mean(RESULT.MAEList, axis=0)
        STORE.MAEList_std = np.std(RESULT.MAEList, axis=0)
        STORE.F1List_mean = np.mean(RESULT.F1List, axis=0)
        STORE.F1List_std = np.std(RESULT.F1List, axis=0)
        STORE.ALC_MZE_mean = np.mean(RESULT.ALC_MZE)
        STORE.ALC_MZE_std = np.std(RESULT.ALC_MZE)
        STORE.ALC_MAE_mean = np.mean(RESULT.ALC_MAE)
        STORE.ALC_MAE_std = np.std(RESULT.ALC_MAE)
        STORE.ALC_F1_mean = np.mean(RESULT.ALC_F1)
        STORE.ALC_F1_std = np.std(RESULT.ALC_F1)
        STORE.ALC_MZE_list = RESULT.ALC_MZE
        STORE.ALC_MAE_list = RESULT.ALC_MAE
        STORE.ALC_F1_list = RESULT.ALC_F1

        sheet_names = ["MZE_mean", "MZE_std", "MAE_mean", "MAE_std", "F1_mean", "F1_std",
                       "ALC_MZE_list","ALC_MAE_list","ALC_F1_list",
                       "ALC_MZE", "ALC_MAE", "ALC_F1","Redun"]
        workbook = xlwt.Workbook()
        for sn in sheet_names:
            print("sn::",sn)
            sheet = workbook.add_sheet(sn)
            n_col = len(STORE.MZEList_mean)
            if sn == "MZE_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MZEList_mean[j - 1])
            elif sn == "MZE_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MZEList_std[j - 1])
            elif sn == "MAE_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MAEList_mean[j - 1])
            elif sn == "MAE_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MAEList_std[j - 1])
            elif sn == "F1_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.F1List_mean[j - 1])
            elif sn == "F1_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.F1List_std[j - 1])

            # ---------------------------------------------------
            elif sn == "ALC_MZE_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_MZE_list) + 1):
                    sheet.write(0,j,STORE.ALC_MZE_list[j - 1])
            elif sn == "ALC_MAE_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_MAE_list) + 1):
                    sheet.write(0,j,STORE.ALC_MAE_list[j - 1])
            elif sn == "ALC_F1_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_F1_list) + 1):
                    sheet.write(0,j,STORE.ALC_F1_list[j - 1])
            # -----------------
            elif sn == "ALC_MZE":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_MZE_mean)
                sheet.write(0, 2, STORE.ALC_MZE_std)
            elif sn == "ALC_MAE":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_MAE_mean)
                sheet.write(0, 2, STORE.ALC_MAE_std)
            elif sn == "ALC_F1":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_F1_mean)
                sheet.write(0, 2, STORE.ALC_F1_std)
            elif sn == "Redun":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.Redun_mean)
                sheet.write(0, 2, STORE.Redun_std)


        save_path = Path(r"E:\EMOC_Batch\BatchSize3\EMOC_LT")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)

