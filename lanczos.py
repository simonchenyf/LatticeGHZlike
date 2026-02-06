"""
File: lanczos.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Lanczos backend for Lanczos-based evolution, occupation
    calculation and parity oscillation.

License:
MIT License

"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import subLanczos, subLanczosJ0, zeroize, getPred, getSz

class Lanczos:
    
    def __init__(self, J0, m, bound):
        self.m = m
        self.bound = bound
        self.x = np.zeros((m,), dtype=np.float64)
        self.y = np.zeros((m-1,), dtype=np.float64)
        self.J0 = J0
        self.N = J0.shape[0]
        self.dim = 2**self.N
        self.V = np.zeros((m, self.dim),dtype=np.complex128)
        self.v1 = np.zeros((self.dim,),dtype=np.complex128)
        self.w = np.zeros((self.dim,),dtype=np.complex128)
        self.V[-1,-1]+=1
        self.V[-1,-1]-=1

    def flip(self, v0, alongx, theta):

        N = self.N
        m = self.m
        bound = self.bound
        x = self.x
        y = self.y
        V = self.V
        v1 = self.v1
        w = self.w

        x[:] = 0
        y[:] = 0
        zeroize.apply(N, m, V, v1, w)

        beta = np.linalg.norm(v0)

        for i in range(m):

            x[i] =  subLanczos.apply(N, beta, v0, v1, w, V[i], alongx)

            beta = np.linalg.norm(w)

            if i < (m-1):
                y[i] = beta
            if beta < bound:
                break

            v0, v1, w = w, v0, v1

        val, vec = eigh_tridiagonal(x[:i+1],y[:i])

        U = (vec.T).dot(V[:i+1,:])

        psi_dt = (np.exp(-1.j*val*theta) * np.conj(vec[0, :].T)).dot(U)

        return psi_dt

    def step(self, v0, alongx, dt):

        J0 = self.J0
        N = self.N
        m = self.m
        bound = self.bound
        x = self.x
        y = self.y
        V = self.V
        v1 = self.v1
        w = self.w

        x[:] = 0
        y[:] = 0
        zeroize.apply(N, m, V, v1, w)

        beta = np.linalg.norm(v0)

        for i in range(m):

            x[i] =  subLanczosJ0.apply(N, beta, J0, v0, v1, w, V[i], alongx)

            beta = np.linalg.norm(w)

            if i < (m-1):
                y[i] = beta
            if beta < bound:
                break

            v0, v1, w = w, v0, v1

        val, vec = eigh_tridiagonal(x[:i+1],y[:i])
        
        U = (vec.T).dot(V[:i+1,:])

        psi_dt = (np.exp(-1.j*val*dt) * np.conj(vec[0, :].T)).dot(U)

        return psi_dt

def getSzDiagonal(Nspins):
    assert Nspins <= 30, 'Nspins>30 not supported in C backend! (only support N of int=int32 type)'
    out = np.zeros(2**Nspins, dtype=np.float64)
    getSz.apply(Nspins, out)
    return out

class Prediction:

    def __init__(self, N, dtype):
        self.N = N
        self.Sz = getSzDiagonal(N)
        self.out = np.zeros(9, dtype=np.float64)
        self.SxPsi = np.zeros(2**N, dtype=dtype)
        self.SyPsi = np.zeros(2**N, dtype=dtype)

    def get_pred(self, psi):

        N = self.N
        Sz = self.Sz
        out = self.out
        SxPsi = self.SxPsi
        SyPsi = self.SyPsi
        getPred.apply(N, psi, Sz, SxPsi, SyPsi, out)
        
        return out