import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from tqdm import tqdm
import os

class DiskMatrix:
    def __init__(self, path, N_used=None, N_multi=10):
        self.path = path
        self.N_multi=N_multi
        assert os.path.exists(path+'_0.npy'), "DiskMatrix does not exist"
        if N_used is None:
            i = 0
            used_num = 0
            while os.path.exists(path+'_%d.npy'%i):
                cur_block = np.load(path+'_%d.npy'%i)
                used_num += cur_block.shape[0]
                i += 1
            self.shape = (used_num, cur_block.shape[1])
        else:
            block = np.load(path+'_0.npy')
            self.shape = (N_used, block.shape[1])

    def dot(self, X):
        return self.rightdot(X)

    def rightdot(self, X, verbose=True):
        assert self.shape[1] == X.shape[0]
        M, N = self.shape
        N, R = X.shape

        rets = []
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            #print (used_num)
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            rets.append(agg_block.dot(X))
            agg_block = []
        if verbose:
            pbar.close()
        return np.concatenate(rets, axis=0)

    def leftdot(self, X, verbose=True):
        assert self.shape[0] == X.shape[1]
        M, N = self.shape
        R, M = X.shape

        rets = 0.0
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            #print (used_num)
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            #print (X[:,used_num-agg_block.shape[0]:used_num].shape)
            #print (agg_block.shape)
            rets = rets + X[:,used_num-agg_block.shape[0]:used_num].dot(agg_block)
            agg_block = []
        if verbose:
            pbar.close()
        return rets

    def norm(self, verbose=True):
        M, N = self.shape

        norm2 = 0.0
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            norm2 = norm2 + np.linalg.norm(agg_block)**2
            agg_block = []
        if verbose:
            pbar.close()

        return np.sqrt(norm2)

def mult(A, B):
    if isinstance(A, DiskMatrix):
        assert isinstance(B, np.ndarray)
        return A.rightdot(B)
    elif isinstance(B, DiskMatrix):
        return B.leftdot(A)
    else:
        return A.dot(B)

def gen_topK_colspace(A, k, n_iter=1):
    # Input
    # A - an (m*n) matrix
    # k - rank
    # n_iter - numer of normalized power iterations
    # Output
    # Q: an (k*n) matrix 

    import time
    t_cur = time.time()
    (m, n) = A.shape

    if (True):
        #Q = np.random.uniform(low=-1.0, high=1.0, size=(k, m)).dot(A).T
        Q = mult(np.random.uniform(low=-1.0, high=1.0, size=(k, m)), A).T
        print (time.time() - t_cur)
        t_cur = time.time()
        Q, _ = lu(Q, permute_l=True)
        print (time.time() - t_cur)
        t_cur = time.time()
        for it in range(n_iter):
            #Q = A.dot(Q)
            Q = mult(A, Q)
            print (time.time() - t_cur)
            t_cur = time.time()
            Q, _ = lu(Q, permute_l=True)
            print (time.time() - t_cur)
            t_cur = time.time()
            #Q = Q.T.dot(A).T
            Q = mult(Q.T, A).T
            print (time.time() - t_cur)
            t_cur = time.time()
            if it + 1 < n_iter:
                (Q, _) = lu(Q, permute_l=True)
            else:
                (Q, _) = qr(Q, mode='economic')
            print (time.time() - t_cur)
            t_cur = time.time()
    else:
        raise NotImplementedError()
    print ("DONE")
    return Q.T


class PCAGenerator:
    # if the input dimension is too large (e.g., ImageNet), we should set approx=True to use randomized PCA
    def __init__(self, N_b, X_shape=None, batch_size=32, preprocess=None, approx=False, basis_only = False):
        self.N_b = N_b
        self.X_shape = X_shape
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.basis = None
        self.approx = approx
        self.basis_only = basis_only

    def fit(self, X):
        if self.X_shape is None:
            raise RuntimeError("X_shape must be passed")
        assert len(X.shape) == 2
        N = X.shape[0]
        if self.approx:
            print ("Using approx pca")
            #import fbpca
            #U, S, Vt = fbpca.pca(A=X, k=self.N_b, raw=True)
            Vt = gen_topK_colspace(A=X, k=self.N_b)
            self.basis = Vt
        else:
            from sklearn.decomposition import PCA
            model = PCA(self.N_b)
            model.fit(X)
            self.basis = model.components_
        if self.basis_only:
            self.basis = self.basis.reshape(self.basis.shape[0], *self.X_shape)

    def save(self, path):
        np.save(path, self.basis.reshape(self.N_b, *self.X_shape))

    def load(self, path):
        self.basis = np.load(path)
        self.X_shape = self.basis.shape[1:]
        if self.basis_only:
            self.basis = self.basis
        else:
            self.basis = self.basis.reshape(self.basis.shape[0], -1)

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
        
        if self.basis is None:
            raise RuntimeError("Must fit or load the model first")

        #ps = []
        #for _ in range(N):
        #    #rv = np.random.randn(self.N_b, 1,1,1)
        #    #pi = (rv * self.basis).sum(axis=0)
        #    #ps.append(pi)
        #    rv = np.random.randn(1, self.N_b)
        #    pi = rv @ self.basis
        #    ps.append(pi)
        ##ps = np.stack(ps, axis=0)
        #ps = np.concatenate(ps, axis=0).reshape(N, *self.X_shape)
        import time
        if self.basis_only:
            rv = np.random.randint(self.N_b, size=(N,))
            ps = self.basis[rv]
        else:
            rv = np.random.randn(N, self.N_b)
            ps = rv.dot(self.basis).reshape(N, *self.X_shape)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp, factor=4.0):
        all_cos2 = 0.0
        for vi in self.basis:
            #cosi = (vi*gt).sum() / np.sqrt( (vi**2).sum() * (gt**2).sum() )
            cosi = (vi.reshape(*self.X_shape)*gt).sum() / np.sqrt( (vi**2).sum() * (gt**2).sum() )
            all_cos2 += (cosi ** 2)
        rho = np.sqrt(all_cos2)
        return rho
