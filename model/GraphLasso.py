import numpy as np
import pandas as pd

class GraphLasso:

    def __init__(self, multiplier, batch_size, max_iter_outer, max_iter_inner, eps):
        self.batch_size = batch_size
        self.multipiler = multiplier
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.eps = eps

    def estimate(self, data : np.ndarray):
        # first check whether the size of the data match
        if data.shape[0] != self.batch_size:
            self.batch_size = data.shape[0]
        # get empirical covariance
        S = np.cov(data.transpose())
        W = S.copy() + self.multipiler * np.ones(shape=(data.shape[1], data.shape[1]), dtype=np.float32)
        P = np.zeros_like(S, dtype=np.float32)
        iter_i = 0
        Noff = data.shape[1] * (data.shape[1] - 1) / 2
        Smag = np.sum(np.triu(S)) / Noff
        W0 = W.copy()
        dW = np.finfo(np.float32).max
        # do this until convergence
        while dW > self.eps and iter_i < self.max_iter_outer:
            iter_i = iter_i + 1
            # iterate through columns
            for i in range(data.shape[1]):
                slice_i = [elem for elem in range(data.shape[1])]
                slice_i = np.delete(arr=slice_i, obj=i, axis=0)
                # approximate covariance
                w_11 = W[slice_i, :]
                w_11 = w_11[:, slice_i]
                w_12 = W[i, slice_i]
                w_12 = np.reshape(w_12, newshape=(data.shape[1] - 1, 1))
                w_22 = W[i, i]
                # exact covariance
                s_12 = S[i, slice_i]
                s_12 = np.reshape(s_12, newshape=(data.shape[1] - 1, 1))
                s_22 = S[i, i]
                # now iterate till convergence
                V = w_11.copy()
                B = np.zeros(shape=(data.shape[1] - 1, 1), dtype=np.float32)
                dB = np.finfo(np.float32).max
                iter_j = 0
                while iter_j < self.max_iter_inner and dB > self.eps:
                    B0 = B.copy()
                    for j in range(data.shape[1] - 1):
                        # compute Beta_j
                        slice_j = [i for i in range(data.shape[1] - 1)]
                        slice_j = np.delete(arr=slice_j, obj=j, axis=0)
                        v_kj = V[slice_j, j]
                        b_k = B[slice_j]
                        res = s_12[j] - np.matmul(v_kj, b_k)
                        B[j, 0] = np.sign(res) * np.max([np.abs(res) - self.multipiler, 0]) / V[j, j]
                    iter_j = iter_j + 1
                    dB = np.mean(np.abs(B - B0)) / (np.mean(np.abs(B0)) + 1e-16)

                w_12 = np.reshape(np.matmul(w_11, B), newshape=(data.shape[1] - 1,))
                W[slice_i, i] = w_12
                W[i, slice_i] = w_12

                p_22 = np.max([0, 1.0 / (w_22 - np.matmul(w_12, B))])
                p_12 = np.reshape(-B * p_22, newshape=(data.shape[1] - 1,))
                P[slice_i, i] = p_12
                P[i, slice_i] = p_12
                P[i, i] = p_22

            dW = np.sum(np.abs(np.triu(W) - np.triu(W0)) / Noff) / Smag

        return P











