import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

from utils.util import kitaev_ham

# parameter definitions
Nsite = 100
Nprime = 2 * Nsite
site_number = 17
e_threshold = 1E-6
params = {
    't': 2.0,
    'Delta': 2.0,
    'mu': 1.0

}

# data processing
w, v = LA.eig(kitaev_ham(Nprime,Nsite, params))
w = np.sort(np.real(w))

# data visualization
plt.scatter(np.arange(0, Nprime), w, color='k')
plt.show()
