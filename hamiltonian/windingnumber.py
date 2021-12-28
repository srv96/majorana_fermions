#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:39:45 2021

@author: gaurab
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA

class hamiltonian:
    def __init__(self, d, l, M):
        self.d, self.l, self.M = d, l, M
        
    def matrix(self,kx,ky,kz):
        return np.array([
            [complex(math.cos(kx)+math.cos(ky)+math.cos(kz)-self.M,0),complex(0,0),complex(self.l*math.sin(kz),self.d),complex(self.l*math.sin(kx),-self.l*math.sin(ky))],
            [complex(0,0),complex(math.cos(kx)+math.cos(ky)+math.cos(kz)-self.M,0),complex(self.l*math.sin(kx),self.l*math.sin(ky)),complex(-self.l*math.sin(kz),self.d)],
            [complex(self.l*math.sin(kz),self.d),complex(self.l*math.sin(kx),-self.l*math.sin(ky)),complex(self.M-math.cos(kx)-math.cos(ky)-math.cos(kz),0),complex(0,0)],
            [complex(self.l*math.sin(kx),self.l*math.sin(ky)),complex(-self.l*math.sin(kz),self.d),complex(0,0),complex(self.M-math.cos(kx)-math.cos(ky)-math.cos(kz),0)]])
    
    
kx=np.arange(-np.pi,np.pi,0.05)
ky=np.arange(-np.pi,np.pi,0.05)
kz=np.arange(-np.pi,np.pi,0.05)
Eigen=[]


if __name__ == "__main__":
    d = 0.5
    l = 1.0
    M = 3
    
    
   
#    result = H.matrix(kx, ky,kz)


H = hamiltonian(d, l, M)
for i in kx:
  for j in ky:
    for k in kz:
       a,b=LA.eig(H.matrix(i,j,k))
       Eigen.append(a[0])
       Eigen.append(a[1])
       Eigen.append(a[2])
       Eigen.append(a[3])
       
       
             
yy = np.real(Eigen)
xx = np.imag(Eigen)
plt.scatter(xx,yy,s=3)
plt.show()
    

