import numpy as np


def kitaev_ham(Nprime, Nsites, params):
    Hmat = np.zeros([Nprime, Nprime])
    Jx = 0.5 * (params['t'] - params['Delta'])
    Jy = 0.5 * (params['t'] + params['Delta'])

    for n in range(Nsites - 1):
        Hmat[2 * n, 2 * n + 1] = Jx
        Hmat[2 * n + 1, 2 * n] = -Jx
        Hmat[2 * n - 1, 2 * n + 2] = -Jy
        Hmat[2 * n + 2, 2 * n - 1] = Jy
        Hmat[2 * n - 1, 2 * n] = params['mu']
        Hmat[2 * n, 2 * n - 1] = -params['mu']

    Hmat[2 * (Nsites - 1) - 1, 2 * (Nsites - 1)] = params['mu']
    Hmat[2 * (Nsites - 1), 2 * (Nsites - 1) - 1] = -params['mu']
    Hmat = 1j * Hmat

    return Hmat
