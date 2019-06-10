import pandas as pd
from sklearn.decomposition import FastICA
import numpy as np
from munkres import Munkres
from copy import deepcopy


def nzdiaghungarian(w):
    S = 1 / np.abs(w)
    indexes = np.vstack(Munkres().compute(deepcopy(S)))
    # Sort by row indices if not already sorted
    indexes = indexes[np.argsort(indexes[:, 0]), :]
    # r-th row moves to `ixs[r]`.
    indexes_perm = indexes[:, 1]

    # w_perm = np.zeros_like(w)
    # w_perm[indexes_perm] = w

    Pr = np.eye(len(w))[:, indexes_perm]
    return np.matmul(Pr, w)


# def slttestperm(b, rows_tol=1e-12):
#     assert (len(b) >= 1 and rows_tol > 0)

def slttestperm(b_i, rows_tol=1e-12):
    """Permute rows and cols of the given matrix.
    """
    n = b_i.shape[0]
    remnodes = np.arange(n)
    b_rem = deepcopy(b_i)
    p = list()

    for i in range(n):
        # Find the row with all zeros
        ixs = np.where(np.sum(np.abs(b_rem), axis=1) < rows_tol)[0]

        if len(ixs) == 0:
            # If empty, return None
            return None
        else:
            # If more than one, arbitrarily select the first
            ix = ixs[0]
            p.append(remnodes[ix])

            # Remove the node
            # remnodes = np.hstack((remnodes[:ix], remnodes[(ix + 1):]))
            remnodes = np.delete(remnodes, ix)

            # Remove  the row and column from b_rem
            # ixs = np.hstack((np.arange(ix), np.arange(ix + 1, len(b_rem))))
            # b_rem = b_rem[ixs, :]
            # b_rem = b_rem[:, ixs]
            b_rem = np.delete(b_rem, (ix), axis=0)
            b_rem = np.delete(b_rem, (ix), axis=1)

    return np.array(p)


def sltscore(B):
    return np.power(np.sum(B[np.triu_indices(len(B))]), 2)


def sltprune(b):
    n = b.shape[0]
    ind = np.argsort(np.abs(b).ravel())

    for i in range(int(n * (n + 1) / 2) - 1, (n * n) - 1):
        # Bi := B, with the i smallest (in absolute value) coefficients to zero
        bi = deepcopy(b)
        bi.ravel()[ind[:i]] = 0
        ixs_perm = slttestperm(bi)

        if ixs_perm is not None:
            b_opt = deepcopy(b)
            b_opt = b_opt[np.ix_(ixs_perm, ixs_perm)]

            # b_opt = b_opt[ixs_perm, :]
            # b_opt = b_opt[:, ixs_perm]
            return b_opt, ixs_perm


def estlingam(data, random_state):
    p = data.shape[1]
    ica = FastICA(max_iter=5000).fit(data)
    w = np.linalg.pinv(ica.mixing_)

    # hungarian algorithm
    w_perm = nzdiaghungarian(w)

    # Divide each row of wp by the diagonal element
    w_perm = w_perm / np.diag(w_perm)[:, np.newaxis]

    # Estimate b
    b_est = np.eye(p) - w_perm

    # permute rows and columns of B so as to get an
    # approximately strictly lower triangular matrix
    Bestcausal, causalperm = sltprune(b_est)

    # Set the upper triangular to zero
    Bestcausal = np.tril(Bestcausal, -1)

    # Permute b_csl back to the original variable
    b_est2 = Bestcausal.copy()
    b_est = Bestcausal  # just rename here
    icausal = iperm(causalperm)

    # b_est[causalperm, :] = deepcopy(b_est)
    # b_est[:, causalperm] = deepcopy(b_est)

    # print(b_est == b_est2[np.ix_(icausal, icausal)])
    return b_est2[np.ix_(icausal, icausal)], causalperm


def iperm(p):
    return np.sort(p)


def prune(X, k, prunefactor=1, npieces=10):
    X = X.values
    X = X.astype(float)
    ndata = X.shape[1]
    p = X.shape[0]
    X_k = X[k, :]
    ik = iperm(k)

    piecesize = np.floor(ndata / npieces)
    Bpieces = np.zeros((p, p, npieces))
    diststdpieces = np.zeros((p, npieces))
    cpieces = np.zeros((p, npieces))
    Bfinal = np.zeros((p, p))
    I_p = np.eye(p)

    for i in range(npieces):
        Xp = X_k[:, int(i * piecesize):int((i + 1) * piecesize)]
        Xpm = np.mean(Xp, axis=1).reshape((len(Xp), 1))
        Xp -= Xpm

        C = np.matmul(Xp, Xp.T) / Xp.shape[1]

        # Hack
        C = C + np.eye(len(C)) * 1e-10
        while np.min(np.linalg.eigvals(C)) < 0:
            C = C + np.eye(len(C)) * 1e-10

        L = tridecomp(invsqrtm(C))

        diag_L = L.diagonal()
        newestdisturbancestd = 1 / np.abs(diag_L)

        L = L / diag_L

        Bnewest = I_p - L

        cnewest = np.matmul(L, Xpm)

        Bnewest = Bnewest[np.ix_(ik, ik)]
        newestdisturbancestd = newestdisturbancestd[ik]
        cnewest = cnewest[ik]

        Bpieces[:, :, i] = Bnewest
        diststdpieces[:, i] = newestdisturbancestd
        cpieces[:, i] = cnewest.ravel()

    for i in range(p):
        Bp_i = Bpieces[i, :, :]
        for j in range(p):
            themean = np.mean(Bp_i[j, :])
            thestd = np.std(Bp_i[j, :])
            if np.abs(themean) < prunefactor * thestd:
                Bfinal[i, j] = 0
            else:
                Bfinal[i, j] = themean
    return Bfinal


def tridecomp(W, choice='ql', only_B=True):
    m = W.shape[0]
    n = W.shape[1]
    Jm = np.flip(np.eye(m), axis=0)
    Jn = np.flip(np.eye(n), axis=0)

    q, r = np.linalg.qr(np.matmul(np.matmul(Jm, W), Jn))
    return np.matmul(np.matmul(Jm, r), Jn)


def invsqrtm(A):
    values, vectors = np.linalg.eig(A)
    lam_m_5 = 1 / np.sqrt(values)
    return np.matmul(np.multiply(vectors, np.tile(lam_m_5, (len(A), 1))), vectors.T)


def lingam(data, only_perm=False, fastICA_tol=1e-14, pmax_nz_brute=8, pmax_slt_brute=8, verbose=False, random_state=1):
    best, k = estlingam(data, random_state)
    return prune(data.T, k)


if __name__ == '__main__':
    file = 'datasets/gmD.csv'
    # file = 'datasets/BD Cont.csv'
    # file = 'datasets/BD Disc.csv'
    # file = 'datasets/BD5 Cluster X Disc Y Outcome (2).csv'
    # file = 'datasets/BD5 Cluster X2 Cont X1 Outcome (1).csv'
    # file = 'datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv'
    # file = 'datasets/ID1 Disc (1).csv'
    # file = 'datasets/ID1 Disc (2).csv'
    # file = 'datasets/mdata.csv'
    # file = 'datasets/mdata2.csv'
    # file = 'datasets/dataset1-continuous.csv'
    # file = 'C:/Users/gaoan/Desktop/dataset/dataset1 (0-5).csv'
    data = pd.read_csv(file)
    # print(data.columns)
    p = lingam(data, True)
    print(p)
