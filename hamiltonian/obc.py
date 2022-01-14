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
    kx = np.arange(-np.pi, np.pi, 1)
    ky = np.arange(-np.pi, np.pi, 1)
    n = 10

    H = hamiltonian(b, d, l, M)

    w_real_all, w_imag_all, v4_all = [], [], []
    counter, star_count = 0, 0

    for i in kx:
        print("\nkx = ", counter, " started")
        counter += 1
        star_count = 101

        for j in ky:
            if star_count < 100:
                print("*", end="")
                star_count += 1
            else:
                print("\n*", end="")
                star_count += 1

            w, v = LA.eig(H.get_matrix(n, i, j))
            v4 = []
            for vi in v:
                vt = vi.reshape(1, vi.shape[0])
                v4.append(np.square(np.real(vt.dot(vt.conjugate().T)[0])[0]))
            v4 = np.array(v4)
            w_real_all.append(np.array([c.real for c in w]))
            w_imag_all.append(np.array([c.imag for c in w]))
            v4_all.append(v4)

    w_real_all = np.array(w_real_all).flatten()
    w_imag_all = np.array(w_imag_all).flatten()
    v4_all = np.array(v4_all).flatten()

    # result = H.get_matrix(n, kx, ky)

    # print(result)

# w, v = LA.eig(H.get_matrix(n, kx, ky))
# w1 = np.real(w)
# w2 = np.imag(w)
#
# v4 = []
# for vi in v:
#     vt = vi.reshape(1, vi.shape[0])
#     v4.append(np.square(np.real(vt.dot(vt.conjugate().T)[0])[0]))
# v4 = np.array(v4)
#
# print(w1.shape, w2.shape, v4.shape)
#
#
# plt.show()

plt.figure(figsize=(27, 18), dpi=1000)
plt.title("dispersion plot")
plt.xlabel('> ===== imaginary  ====> ', fontsize=12)
plt.ylabel('> ===== real  ====> ', fontsize=12)
plt.scatter(w_imag_all, w_real_all, marker='.', s=0.01, linewidths=4, c=v4_all, cmap=plt.cm.coolwarm)
plt.savefig('filename.png', dpi=1000)
plt.show()
