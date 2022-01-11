# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA


class hamiltonian:
    def __init__(self, b, d, l, M):
        self.b, self.d, self.l, self.M = b, d, l, M

        self.t = 0.5 * np.array([[complex(1, 0), complex(0, 1), complex(0, 0), complex(0, 0)],
                                 [complex(0, 1), complex(-1, 0),
                                  complex(0, 0), complex(0, 0)],
                                 [complex(0, 0), complex(0, 0),
                                  complex(1, 0), complex(0, -1)],
                                 [complex(0, 0), complex(0, 0), complex(0, -1), complex(-1, 0)]])
        self.tt = 0.5 * np.array([[complex(1, 0), complex(0, -1), complex(0, 0), complex(0, 0)],
                                  [complex(0, -1), complex(-1, 0),
                                   complex(0, 0), complex(0, 0)],
                                  [complex(0, 0), complex(0, 0),
                                   complex(1, 0), complex(0, 1)],
                                  [complex(0, 0), complex(0, 0), complex(0, 1), complex(-1, 0)]])

    def get_E(self, kx, ky):
        return np.array(
            [[complex(math.cos(kx) + math.cos(ky) - self.M + self.b, 0), complex(0, self.d), complex(self.b, -self.b),
              self.l * complex(math.sin(kx), -math.sin(ky))],
             [complex(0, self.d), complex(-math.cos(kx) - math.cos(ky) + self.M + self.b, 0),
              self.l * complex(math.sin(kx), math.sin(ky)), complex(self.b, -self.b)],
             [complex(self.b, self.b), self.l * complex(math.sin(kx), -math.sin(ky)),
              complex(math.cos(kx) + math.cos(ky) - self.M - self.b, 0), complex(0, self.d)],
             [self.l * complex(math.sin(kx), math.sin(ky)), complex(self.b, self.b), complex(0, self.d),
              complex(-math.cos(kx) - math.cos(ky) + self.M - self.b, 0)]])

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




if __name__ == "__main__":
    b = 0.0
    d = 1
    l = 1
    M = 3
    kx = 0
    ky = 0
    n = 800

    H = hamiltonian(b, d, l, M)

    result = H.get_matrix(n, kx, ky)

    print(result)

w, v = LA.eig(H.get_matrix(n, kx, ky))
w1 = np.real(w)
w2 = np.imag(w)

plt.scatter(w1, w2, s=10)
plt.plot
