#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:22:05 2021

@author: gaurab
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA

class hamiltonian:
    def __init__(self, b, d, l, M, o):
        self.b, self.d, self.l, self.M , self.o= b, d, l, M, o
        
        self.t = 0.5*np.array([[complex(1,0),complex(0,0),complex(0,self.l),complex(0,0)],
                               [complex(0,0),complex(1,0), complex(0,0),complex(0,-self.l)],
                               [complex(0,self.l),complex(0,0),complex(-1,0),complex(0,0)],
                               [complex(0,0),complex(0,-self.l),complex(0,0),complex(-1,0)]])
        
        self.tt = 0.5*np.array([[complex(1,0),complex(0,0),complex(0,-self.l),complex(0,0)],
                                [complex(0,0),complex(1,0), complex(0,0),complex(0,self.l)],
                                [complex(0,-self.l),complex(0,0),complex(-1,0),complex(0,0)],
                                [complex(0,0),complex(0,self.l),complex(0,0),complex(-1,0)]])
        
    def get_E(self, kx, ky):
        return np.array([
            [complex(self.b*math.sin(o)+self.b*math.cos(o)+math.cos(kx)+math.cos(ky)-self.M,0),complex(self.b*(math.sin(o)+math.cos(o)),-self.b*(math.sin(o)+math.cos(o))),complex(0,self.d),complex(self.l*math.sin(kx),-self.l*math.sin(ky))],
            [complex(self.b*(math.sin(o)+math.cos(o)),self.b*(math.sin(o)+math.cos(o))),complex(-self.b*math.sin(o)-self.b*math.cos(o)+math.cos(kx)+math.cos(ky)-self.M,0),complex(self.l*math.sin(kx),-self.l*math.sin(ky)),complex(0,-self.d)],
            [complex(o,self.d),complex(self.l*math.sin(kx),-self.l*math.sin(ky)),complex(self.b*math.sin(o)-self.b*math.cos(o)-math.cos(kx)-math.cos(ky)+self.M,0),complex(self.b*(math.sin(o)-math.cos(o)),-self.b*(math.sin(o)-math.cos(o)))],
            [complex(self.l*math.sin(kx),self.l*math.sin(ky)),complex(0,-self.d),complex(self.b*(math.sin(o)-math.cos(o)),-self.b*(-math.sin(o)+math.cos(o))),complex(-self.b*math.sin(o)+self.b*math.cos(o)-math.cos(kx)-math.cos(ky)+self.M,0)]])
    
    
    def get_matrix(self, n, kx, ky):
        e = self.get_E(kx, ky)
        e_matrix = np.zeros([4 * n, 4 * n], dtype=complex)
        for i in range(n):
            for j in range(4):
                for k in range(4):
                    e_matrix[4 * i + j][4 * i + k] = e[j][k]
            if i < n - 1:
                for j in range(4):
                    for k in range(4):
                        e_matrix[4 * i + j][4 * (i + 1) + k] = self.t[j][k]
                for j in range(4):
                    for k in range(4):
                        e_matrix[4 * (i + 1) + j][4 * i + k] = self.tt[j][k]
        return e_matrix
    
kx=np.arange(-np.pi,np.pi,0.01)
ky=np.arange(-np.pi,np.pi,0.01)
Eigen=[]
if __name__ == "__main__":
    b = 0.0
    d = 1.0
    l = 1.0
    M = 3.0
    
    o = 0.0
    n = 10

    H = hamiltonian(b, d, l, M,o)
    for i in kx:
        for j in ky:
            w, v = LA.eig(H.get_matrix(n, i, j))
            Eigen.append(w)

    result = H.get_matrix(n, kx, ky)

    print(result)




w1 = np.real(Eigen)
w2 = np.imag(Eigen)

plt.scatter(w2 ,w1, s=3)
plt.show()