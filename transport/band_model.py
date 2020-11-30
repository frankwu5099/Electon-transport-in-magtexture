from numpy import inf, nan
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.ndimage.interpolation import shift
import sys, time, os
import pandas as pd
import datetime
import numba
import seaborn as sns
import itertools
import json
from scipy.sparse import coo_matrix

sigma0 = np.array([[1,0],[0,1]]).astype(complex)
sigmax = np.array([[0,1],[1,0]]).astype(complex)
sigmay = np.array([[0. + 0.j, 0. - 1.j],[0. + 1.j, 0. + 0.j]]).astype(complex)
sigmaz = np.array([[1,0],[0,-1]]).astype(complex)
class ElectronMagModel:
    def __init__(self, mconfig, lambda_,  J, Bz):
        _, self.Lx, self.Ly, self.Lz = mconfig.shape
        self.Nspins = self.Lx * self.Ly * self.Lz
        self.Ndim = self.Nspins * 2
        self.mx, self.my, self.mz = mconfig[0], mconfig[1], mconfig[2]
        self.lambda_ = lambda_
        self.J = J
        self.Bz = Bz
        self.make_Ham_bases()

    def make_Ham_bases(self):
        ham_kx_forward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_kx_backward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_ky_forward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_ky_backward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_kz_forward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_kz_backward_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        ham_coupling_base = np.zeros([self.Lx, self.Ly, self.Lz, 2,
                        self.Lx, self.Ly, self.Lz, 2]).astype(complex)
        for i, j, k in itertools.product(range(self.Lx), range(self.Ly),  range(self.Lz)):
            if i == self.Lx - 1:
                ham_kx_forward_base[0, j, k, :, i, j, k, :] += \
                    np.exp(-1.j * (self.Bz*self.Lx*j)) * \
                    (-1. * sigma0 + 1.j * self.lambda_ * sigmax)
            else:
                ham_kx_forward_base[i + 1, j, k, :, i, j, k, :] += \
                    (-1. * sigma0 + 1.j * self.lambda_ * sigmax)
            if i == 0:
                ham_kx_backward_base[self.Lx - 1, j, k, :, i, j, k, :] += \
                    np.exp(1.j * (self.Bz*self.Lx*j)) * \
                    (-1. * sigma0 - 1.j * self.lambda_ * sigmax)
            else: 
                ham_kx_backward_base[i - 1, j, k, :, i, j, k, :] += \
                    (-1. * sigma0 - 1.j * self.lambda_ * sigmax)
            ham_ky_forward_base[i, (j+1)%self.Ly, k, :, i, j, k, :] += \
                np.exp(1.j * (self.Bz*i)) * \
                (-1. * sigma0 + 1.j * self.lambda_ * sigmay)#
            ham_ky_backward_base[i, (j+self.Ly-1)%self.Ly, k, :, i, j, k, :] += \
                np.exp(-1.j * (self.Bz*i)) * \
                (-1. * sigma0 - 1.j * self.lambda_ * sigmay)#
            if self.Lz > 1: # only for 3D config
                ham_kz_forward_base[i, j, (k+1)%self.Lz, :, i, j, k, :] += \
                    (-1. * sigma0 + 1.j * self.lambda_ * sigmaz)#
                ham_kz_backward_base[i, j, (k+self.Lz-1)%self.Lz, :, i, j, k, :] += \
                    (-1. * sigma0 - 1.j * self.lambda_ * sigmaz)#
            ham_coupling_base[i, j, k, :, i, j, k, :] += \
                    -self.J * (
                        self.mx[i, j, k] * sigmax + 
                        self.my[i, j, k] * sigmay + 
                        self.mz[i, j, k] * sigmaz
                        )
        self.bases_matrices = {
            "x+" : coo_matrix(ham_kx_forward_base.reshape(self.Ndim, self.Ndim)),
            "x-" : coo_matrix(ham_kx_backward_base.reshape(self.Ndim, self.Ndim)),
            "y+" : coo_matrix(ham_ky_forward_base.reshape(self.Ndim, self.Ndim)),
            "y-" : coo_matrix(ham_ky_backward_base.reshape(self.Ndim, self.Ndim)),
            "z+" : coo_matrix(ham_kz_forward_base.reshape(self.Ndim, self.Ndim)),
            "z-" : coo_matrix(ham_kz_backward_base.reshape(self.Ndim, self.Ndim)),
            "Jm" : coo_matrix(ham_coupling_base.reshape(self.Ndim, self.Ndim))
        }

    def Hamiltonian(self, kx, ky, kz, ):
        H = self.bases_matrices["x+"] * np.exp(-1.j * kx) +\
            self.bases_matrices["x-"] * np.exp(1.j * kx) +\
            self.bases_matrices["y+"] * np.exp(-1.j * ky) +\
            self.bases_matrices["y-"] * np.exp(1.j * ky) +\
            self.bases_matrices["z+"] * np.exp(-1.j * kz) +\
            self.bases_matrices["z-"] * np.exp(1.j * kz) +\
            self.bases_matrices["Jm"]
        
        return H.toarray()

    def Hamiltoniandkx(self, kx, ky, kz):
        Hkx = self.bases_matrices["x+"] * (-1.j * np.exp(-1.j * kx)) +\
            self.bases_matrices["x-"] * (1.j * np.exp(1.j * kx))
        return Hkx.toarray()

    def Hamiltoniandky(self, kx, ky, kz):
        Hky = self.bases_matrices["y+"] * (-1.j * np.exp(-1.j * ky)) +\
            self.bases_matrices["y-"] * (1.j * np.exp(1.j * ky))
        return Hky

    def eigen(self, kx, ky, kz):
        ham = self.Hamiltonian(kx, ky, kz)
        eigenValues, eigenVectors = scipy.linalg.eigh(ham)
        idx = eigenValues.argsort()[:]  
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors
    
    def getks(self, Lkx, Lky, Lkz = 1):
        kxs = np.linspace(-np.pi/self.Lx, np.pi/self.Lx, Lkx+1)[:Lkx] + np.pi/self.Lx/Lkx
        kys = np.linspace(-np.pi/self.Ly, np.pi/self.Ly, Lky+1)[:Lky] + np.pi/self.Ly/Lky
        if self.Lz == 0:
            if Lkz!=1:
                print("invalid parameter")
                raise
            else:
                kzs = np.array([0])
        else:
            kzs = np.linspace(-np.pi/self.Lz, np.pi/self.Lz, Lkz+1)[:Lkz] + np.pi/self.Lz/Lkz
        return kxs, kys, kzs


class BandSolver:
    def __init__(self, model : ElectronMagModel):
        self.model = model
        self.kmesh_ = None
        self.Nbands = model.Ndim
    def solvebands(self, kxs, kys, kzs, eta = 1e-5, statusbar = None):
        self.kxs_, self.kys_, self.kzs_ = kxs, kys, kzs
        len_kx, len_ky, len_kz = len(kxs), len(kys), len(kzs)

        datashape = (len_kx, len_ky, len_kz, self.Nbands)
        Es = np.zeros(datashape, dtype = np.float)
        sigmaxy = np.zeros(datashape, dtype = np.float)
        sigmaxx = np.zeros(datashape, dtype = np.float)
        if statusbar is None:
            iterks = itertools.product(range(len_kz), range(len_ky), range(len_kx))
        else:
            total = len_kz * len_ky * len_kx
            iterks = statusbar(
                itertools.product(range(len_kz), range(len_ky), range(len_kx)), 
                total = total)

        for kzi, kyi, kxi in iterks:
            eigvals, eigvecs = self.model.eigen(kxs[kxi], kys[kyi], kzs[kzi])
            Es[kxi, kyi, kzi] = eigvals
            eigvals = eigvals.reshape(-1, 1)
            dhdkx = self.model.Hamiltoniandkx(kxs[kxi], kys[kyi], kzs[kzi])
            dhdky = self.model.Hamiltoniandky(kxs[kxi], kys[kyi], kzs[kzi])
            ndhdkxm = np.conj((eigvecs.T.conj() @ dhdkx) @ eigvecs)
            ndhdkym = eigvecs.T.conj() @ (dhdky @ eigvecs)
            Kuboxynn = ndhdkxm * ndhdkym
            Kuboxxnn = ndhdkxm * np.conj(ndhdkxm)
            Ediff = eigvals.T - eigvals
            weight1 = 1. / ((Ediff)**2 + eta**2)
            np.fill_diagonal(weight1, 0)
            weight2 = 1. / Ediff / ((Ediff)**2 + eta**2)
            np.fill_diagonal(weight2, 0)
            sigmaxy[kxi, kyi, kzi] =\
                4*np.pi*np.pi/\
                (self.model.Lx * self.model.Ly* (len_kx) * (len_ky))*\
                (-2 * Kuboxynn.imag * weight1).sum(axis = 1)
            sigmaxx[kxi, kyi, kzi] =\
                4*np.pi*np.pi*eta/\
                (self.model.Lx * self.model.Ly* (len_kx) * (len_ky))*\
                (2 * Kuboxxnn * weight2).sum(axis = 1)
        self.result = {
            "sigmaxx": sigmaxx,
            "sigmaxy": sigmaxy,
            "energy": Es,
        }
        return self.result
