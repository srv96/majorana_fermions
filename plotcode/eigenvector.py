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
w, v = LA.eig(kitaev_ham(Nprime, Nsite, params))

# data visualization
plt.plot(np.arange(0, Nprime), (v[:, site_number]), color='k')
plt.bar(np.arange(0, Nprime), (v[:, site_number]))
plt.show()
