import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA


class rotated_hamiltonian:
    def __init__(self, b, d, l, M, o):
        self.b, self.d, self.l, self.M, self.o = b, d, l, M, o

        self.t1 = np.array([[complex(0, 0), complex(0, 0), complex(0.5 * (1 - self.l), 0), complex(0, 0)],
                            [complex(0, 0), complex(0, 0), complex(0, 0), complex(0.5 * (1 + self.l), 0)],
                            [complex(0.5 * (1 + self.l), 0), complex(0, 0), complex(0, 0), complex(0, 0)],
                            [complex(0, 0), complex(0.5 * (1 - self.l), 0), complex(0, 0), complex(0, 0)]])
        self.t2 = np.array([[complex(0, 0), complex(0, 0), complex(0.5 * (1 + self.l), 0), complex(0, 0)],
                            [complex(0, 0), complex(0, 0), complex(0, 0), complex(0.5 * (1 - self.l), 0)],
                            [complex(0.5 * (1 - self.l), 0), complex(0, 0), complex(0, 0), complex(0, 0)],
                            [complex(0, 0), complex(0.5 * (1 + self.l), 0), complex(0, 0), complex(0, 0)]])

    def get_E(self, kx, ky):
        return np.array([[complex(self.b * math.sin(o), 0),
                          complex(self.b * math.sin(o), -self.b * math.sin(0)),
                          complex(self.b * math.cos(o) + self.d + math.cos(kx) + math.cos(ky) - self.M, 0),
                          complex(self.b * math.cos(o) - self.l * math.sin(ky),
                                  -self.b * math.cos(o) - self.l * math.sin(kx))],
                         [complex(self.b * math.sin(o), self.b * math.sin(0)),
                          complex(-self.b * math.sin(o), 0),
                          complex(self.b * math.cos(o) + self.l * math.sin(ky),
                                  self.b * math.cos(o) - self.l * math.sin(kx)),
                          complex(-self.b * math.cos(o) + self.d + math.cos(kx) + math.cos(ky) - self.M, 0)],
                         [complex(self.b * math.cos(o) - self.d + math.cos(kx) + math.cos(ky) - self.M, 0),
                          complex(self.b * math.cos(o) + self.l * math.sin(ky),
                                  -self.b * math.cos(o) + self.l * math.sin(kx)),
                          complex(self.b * math.sin(o), 0),
                          complex(self.b * math.sin(o), -self.b * math.sin(0))],
                         [complex(self.b * math.cos(o) - self.l * math.sin(ky),
                                  self.b * math.cos(o) + self.l * math.sin(kx)),
                          complex(-self.b * math.cos(o) - self.d + math.cos(kx) + math.cos(ky) - self.M, 0),
                          complex(self.b * math.sin(o), self.b * math.sin(0)),
                          complex(-self.b * math.sin(o), 0)]])

    def get_matrix(self, n, kx, ky):
        e1 = self.get_E(kx, ky)
        e1_matrix = np.zeros([4 * n, 4 * n], dtype=complex)
        for i in range(n):
            for j in range(4):
                for k in range(4):
                    e1_matrix[4 * i + j][4 * i + k] = e1[j][k]
            if i < n - 1:
                for j in range(4):
                    for k in range(4):
                        e1_matrix[4 * i + j][4 * (i + 1) + k] = self.t1[j][k]
                for j in range(4):
                    for k in range(4):
                        e1_matrix[4 * (i + 1) + j][4 * i + k] = self.t2[j][k]
        return e1_matrix


if __name__ == "__main__":

    d, l, M, b, o = 1, 1, 3, 0.5, 0.9

    n = 10
    kx = np.arange(-np.pi, np.pi, 0.01)
    ky = np.arange(-np.pi, np.pi, 0.01)

    H1 = rotated_hamiltonian(b, d, l, M, o)

    w_real_all, w_imag_all = [], []

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

            w, v = LA.eig(H1.get_matrix(n, i, j))
            w_real_all.append(np.array([c.real for c in w]))
            w_imag_all.append(np.array([c.imag for c in w]))

    w_real_all = np.array(w_real_all).flatten()
    w_imag_all = np.array(w_imag_all).flatten()

    plt.figure(figsize=(27, 18), dpi=1000)
    plt.title("dispersion plot")
    plt.xlabel('> ===== imaginary  ====> ', fontsize=12)
    plt.ylabel('> ===== real  ====> ', fontsize=12)
    plt.scatter(w_imag_all, w_real_all, s=0.003)
    plt.savefig('filename.png', dpi=1000)
    # plt.show()
