"""============================================================================
Fitting and plotting script for kernel ridge regression (see rffridge.py).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
from   rffridge import RFFRidgeRegression
from   sklearn.gaussian_process.kernels import RBF
from   sklearn.kernel_ridge import KernelRidge


# -----------------------------------------------------------------------------

N     = 100
X     = np.linspace(-10, 10, N)[:, None]
mean  = np.zeros(N)
cov   = RBF()(X.reshape(N, -1))
y     = np.random.multivariate_normal(mean, cov)
noise = np.random.normal(0, 0.5, N)
y    += noise

# Finer resolution for smoother curve visualization.
X_test = np.linspace(-10, 10, N*2)[:, None]

# Set up figure and plot data.
fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 5)
ax1, ax2  = axes
cmap      = plt.cm.get_cmap('Blues')

ax1.scatter(X, y, s=30, c=[cmap(0.3)])
ax2.scatter(X, y, s=30, c=[cmap(0.3)])

# Fit kernel ridege regression using an RBF kernel.
clf    = KernelRidge(kernel=RBF())
clf    = clf.fit(X, y)
y_pred = clf.predict(X_test)
ax1.plot(X_test, y_pred, c=cmap(0.9))

# Fit kernel ridge regression using random Fourier features.
rff_dim = 20
clf     = RFFRidgeRegression(rff_dim=rff_dim)
clf.fit(X, y)
y_pred  = clf.predict(X_test)
ax2.plot(X_test, y_pred, c=cmap(0.9))

# Labels, etc.
ax1.margins(0, 0.1)
ax1.set_title('RBF kernel regression')
ax1.set_ylabel(r'$y$', fontsize=14)
ax1.set_xticks([])
ax2.margins(0, 0.1)
ax2.set_title(rf'RFF ridge regression, $R = {rff_dim}$')
ax2.set_ylabel(r'$y$', fontsize=14)
ax2.set_xlabel(r'$x$', fontsize=14)
ax2.set_xticks(np.arange(-10, 10.1, 1))
plt.tight_layout()
plt.show()
