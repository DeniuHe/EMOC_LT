import numpy as np
from sklearn import datasets
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

class DPC(object):
    def __init__(self, X, clusterNum, distPercent):
        self.X = X
        self.nSample = X.shape[0]
        self.clusterNum = clusterNum
        self.distPercent = distPercent
        self.rho = np.zeros(self.nSample)
        self.delta = np.zeros(self.nSample)
        self.gamma = np.zeros(self.nSample)
        self.leader = np.ones(self.nSample, dtype=int) * int(-1)
        self.distMatrix = pairwise_distances(X,metric="euclidean")
        self.distCut = np.max(self.distMatrix) * (self.distPercent / 100)
        self.rho = np.sum(np.exp(-(self.distMatrix/self.distCut)**2), axis=1)
        self.order_rho = np.flipud(np.argsort(self.rho))


        # -------------密度最大点的delta-------------
        self.delta[self.order_rho[0]] = np.max(self.distMatrix[self.order_rho[0],:])
        self.leader[self.order_rho[0]] = -1
        # -----------获取非密度最大点的delta和leader--------
        for i in range(1, self.nSample):
            min_dist = np.inf
            min_idx = -1
            for j in range(i):
                dist = self.distMatrix[self.order_rho[i], self.order_rho[j]]
                if dist < min_dist:
                    min_dist = dist
                    min_idx = self.order_rho[j]
            self.delta[self.order_rho[i]] = min_dist
            self.leader[self.order_rho[i]] = min_idx
        self.gamma = self.rho * self.delta
        self.order_gamma = np.flipud(np.argsort(self.gamma))

        # --------给聚类中心分配簇标签----------
        self.clusterIndex = np.ones(self.nSample, dtype=int) * (-1)
        for i in range(self.clusterNum):
            self.clusterIndex[self.order_gamma[i]] = i

        for i in range(self.nSample):
            if self.clusterIndex[self.order_rho[i]] == -1:
                self.clusterIndex[self.order_rho[i]] = self.clusterIndex[self.leader[self.order_rho[i]]]



if __name__ == '__main__':
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # X, y = datasets.make_blobs(n_features=2, n_samples=200, centers=3, cluster_std=1.5, random_state=9)
    X, y = datasets.make_blobs(n_features=2, n_samples=200, centers=3, cluster_std=1.5, random_state=10)
    # plt.scatter(X[:,0], X[:,1],marker="o", s=30)
    # plt.show()



    model = DPC(X,clusterNum=3,distPercent=2)
    leader = model.leader
    rho = model.rho
    rho = (rho ** 0.5) * 100
    ord_gamma = model.order_gamma
    cluster_id = model.clusterIndex
    plt.scatter(X[:,0],X[:,1],c=rho,marker="o",cmap='viridis')
    plt.colorbar()
    plt.show()

    # plt.scatter(X[:,0],X[:,1],c=cluster_id)
    plt.scatter(X[:,0],X[:,1])
    # plt.scatter(X[:,0],X[:,1],label="Instances")



    for idx in range(len(y)):
        jdx = leader[idx]
        if jdx != -1:
            plt.plot(X[[idx,jdx]][:,0],X[[idx,jdx]][:,1],c="k",lw=1)

    # plt.scatter(X[ord_gamma[0]][0], X[ord_gamma[0]][1],c='w', edgecolors='r',marker="o",linewidths=3,s=110)
    # plt.scatter(X[ord_gamma[1]][0], X[ord_gamma[1]][1],c='w', edgecolors='r',marker="o",linewidths=3,s=110)
    # plt.scatter(X[ord_gamma[2]][0], X[ord_gamma[2]][1],c='w', edgecolors='r',marker="o",linewidths=3,s=110)

    # plt.scatter(X[ord_gamma[0]][0], X[ord_gamma[0]][1],c='r',marker="*",s=300)
    # plt.scatter(X[ord_gamma[1]][0], X[ord_gamma[1]][1],c='r',marker="*",s=300)
    # plt.scatter(X[ord_gamma[2]][0], X[ord_gamma[2]][1],c='r',marker="*",s=300)
    plt.tight_layout()
    # plt.legend(loc="lower left",prop=font1)
    plt.show()


