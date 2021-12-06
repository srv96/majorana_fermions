import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA


class hamiltonian:
    def __init__(self, b, d, l, M, o):
        self.b, self.d, self.l, self.M, self.o = b, d, l, M, o

        self.tx = 0.5 * np.array([[complex(0, 0), complex(0, 0), complex(1, 0), complex(self.l, 0)],
                                  [complex(0, 0), complex(0, 0), complex(self.l, 0), complex(1, 0)],
                                  [complex(1, 0), complex(-self.l, 0), complex(0, 0), complex(0, 0)],
                                  [complex(-self.l, 0), complex(1, 0), complex(0, 0), complex(0, 0)]])

        self.ttx = 0.5 * np.array([[complex(0, 0), complex(0, 0), complex(1, 0), complex(-self.l, 0)],
                                   [complex(0, 0), complex(0, 0), complex(-self.l, 0), complex(1, 0)],
                                   [complex(1, 0), complex(self.l, 0), complex(0, 0), complex(0, 0)],
                                   [complex(self.l, 0), complex(1, 0), complex(0, 0), complex(0, 0)]])

        self.tz = 0.5 * np.array([[complex(0, 0), complex(0, 0), complex(0.5 * (1 + self.l), 0), complex(0, 0)],
                                  [complex(0, 0), complex(0, 0), complex(0, 0), complex(0.5 * (1 - self.l), 0)],
                                  [complex(0.5 * (1 - self.l), 0), complex(0, 0), complex(0, 0), complex(0, 0)],
                                  [complex(0, 0), complex(0.5 * (1 + self.l), 0), complex(0, 0), complex(0, 0)]])

        self.ttz = 0.5 * np.array([[complex(0, 0), complex(0, 0), complex(0.5 * (1 - self.l), 0), complex(0, 0)],
                                   [complex(0, 0), complex(0, 0), complex(0, 0), complex(0.5 * (1 + self.l), 0)],
                                   [complex(0.5 * (1 + self.l), 0), complex(0, 0), complex(0, 0), complex(0, 0)],
                                   [complex(0, 0), complex(0.5 * (1 - self.l), 0), complex(0, 0), complex(0, 0)]])

    def onsite_matrix(self, ky):
        return np.array([[complex(self.b * (math.sin(0) + math.cos(0)), 0),
                          complex(self.b * (math.sin(0) + math.cos(0)), -self.b * (math.sin(0) + math.cos(0))),
                          complex(math.cos(ky) - self.M + self.d, 0),
                          complex(-self.l * math.sin(ky), 0)],
                         [complex(self.b * (math.sin(0) + math.cos(0)), self.b * (math.sin(0) + math.cos(0))),
                          complex(-self.b * (math.sin(0) + math.cos(0)), 0),
                          complex(-self.l * math.sin(ky), 0),
                          complex(math.cos(ky) - self.M + self.d, 0)],
                         [complex(math.cos(ky) - self.M + self.d, 0),
                          complex(self.l * math.sin(ky), 0),
                          complex(self.b * (math.sin(0) + math.cos(0)), 0),
                          complex(self.b * (math.sin(0) + math.cos(0)), -self.b * (math.sin(0) + math.cos(0)))],
                         [complex(-self.l * math.sin(ky), 0),
                          complex(math.cos(ky) - self.M + self.d, 0),
                          complex(self.b * (math.sin(0) + math.cos(0)), self.b * (math.sin(0) + math.cos(0))),
                          complex(-self.b * (math.sin(0) + math.cos(0)), 0)]])

    def logical_matrix(self, n):
        lm = []
        lm_om, lm_tx, lm_ttx, lm_tz, lm_ttz = "om", "tx", "ttx", "tz", "ttz"
        z_counter = int(np.sqrt(n)) - 1
        counter = 0
        layer_size = int(np.sqrt(n))

        for i in range(n):
            lmr = []
            for j in range(n):
                lmr.append("0")
            lm.append(lmr)

        for i in range(n - 1):
            for j in range(n - 1):
                if i == j:
                    lm[i][j] = lm_om
                    if counter < z_counter:
                        lm[i][j + 1] = lm_ttx
                        lm[i + 1][j] = lm_tx
                        counter = counter + 1
                    else:
                        lm[i][j + 1] = "0"
                        lm[i + 1][j] = "0"
                        counter = 0
                    if i < n - layer_size:
                        lm[i][j + layer_size] = lm_tz
                        lm[i + layer_size][j] = lm_ttz

        return np.array(lm)

    def get_matrix(self, ky, n):
        if np.square(int(np.sqrt(n))) != n:
            print("matrix size is not valid")
        else:
            om = self.onsite_matrix(ky)
            lm = self.logical_matrix(n)
            pm = np.zeros([4 * n, 4 * n], dtype=complex)
            lm_om, lm_tx, lm_ttx, lm_tz, lm_ttz = "om", "tx", "ttx", "tz", "ttz"
            for i in range(n):
                for j in range(n):
                    for k in range(4):
                        for l in range(4):
                            if lm[i][j] == lm_om:
                                pm[i * 4 + k][j * 4 + l] = om[k][l]
                            elif lm[i][j] == lm_ttx:
                                pm[i * 4 + k][j * 4 + l] = self.ttx[k][l]
                            elif lm[i][j] == lm_tx:
                                pm[i * 4 + k][j * 4 + l] = self.ttx[k][l]
                            elif lm[i][j] == lm_tz:
                                pm[i * 4 + k][j * 4 + l] = self.tz[k][l]
                            elif lm[i][j] == lm_ttz:
                                pm[i * 4 + k][j * 4 + l] = self.ttz[k][l]
            return pm


if __name__ == "__main__":
    b = 0.2
    d = 1
    l = 1
    M = 3
    o = 0

    kx = math.pi / 3
    ky = math.pi / 3
    n = 36

    kys = np.arange(-np.pi, np.pi, 0.1)

    H = hamiltonian(b, d, l, M, o)

    for ky in kys:
        result = H.get_matrix(ky, n)
        w, v = np.linalg.eig(result)
        real = [c.real for c in w]
        imaginary = [c.imag for c in w]
        plt.scatter(imaginary, real, s=0.5, color='black')

    plt.show()

# print(result)
