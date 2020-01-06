"""============================================================================
Fitting and plotting script for Gaussian process regression (see rffgpr.py).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
from   rffgpr import RFFGaussianProcessRegressor
from   sklearn.gaussian_process.kernels import RBF
from   sklearn.gaussian_process import GaussianProcessRegressor


# -----------------------------------------------------------------------------

def get_random_function(N, kernel):
    X = np.linspace(0, 50, N)[:, None]
    mean = np.zeros(N)
    cov = kernel(X.reshape(N, -1))
    y = np.random.multivariate_normal(mean, cov)
    # y = np.sin(X.flatten() * 3) + np.random.normal(0, 0.1, N)
    return X, y


def get_data(N, N_train, kernel):
    X_test, y_test = get_random_function(N, kernel)
    inds    = np.random.choice(N, size=N_train, replace=False)
    if 0 not in inds:
        inds[0] = 0
    if N-1 not in inds:
        inds[N_train-1] = N-1
    assert(np.unique(inds).size == inds.size)
    y_train = y_test[inds]
    X_train = X_test[inds]
    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------

N       = 1000
N_train = 200
kernel  = RBF(1.0, length_scale_bounds="fixed")

X_train, y_train, X_test, y_test = get_data(N, N_train, kernel)

# Set up figure and plot data.
fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 5)
ax1, ax2  = axes
cmap      = plt.cm.get_cmap('Blues')

ax1.plot(X_test, y_test, c=cmap(0.9), zorder=1)
ax1.scatter(X_train, y_train, s=40, c=[cmap(0.9)], zorder=2)
ax2.plot(X_test, y_test, c=cmap(0.9), zorder=1)
ax2.scatter(X_train, y_train, s=40, c=[cmap(0.9)], zorder=2)

clf = GaussianProcessRegressor(kernel=kernel)
clf = clf.fit(X_train, y_train)
y_mean, y_cov = clf.predict(X_test, return_cov=True)
ax1.plot(X_test, y_mean, c='r')
# Plot model uncertainty.
y_std = np.sqrt(np.diag(y_cov))
ax1.fill_between(X_test.squeeze(), y_mean-2*y_std, y_mean+2*y_std,
    color='r', zorder=0, alpha=0.1)

# Fit kernel ridge regression using random Fourier features.
rff_dim = 100
clf     = RFFGaussianProcessRegressor(rff_dim=rff_dim)
clf.fit(X_train, y_train)
y_mean, y_cov = clf.predict(X_test)
ax2.plot(X_test, y_mean, c='r', zorder=3)
# Plot model uncertainty.
y_std = np.sqrt(np.diag(y_cov))
ax2.fill_between(X_test.squeeze(), y_mean-2*y_std, y_mean+2*y_std,
    color='r', zorder=0, alpha=0.1)

# Labels, etc.
ax1.margins(0, 0.1)
ax1.set_title('Gaussian process regression')
ax1.set_ylabel(r'$y$', fontsize=14)
ax1.set_xticks([])
ax2.margins(0, 0.1)
ax2.set_title(rf'RFF Gaussian process regression, $R = {rff_dim}$')
ax2.set_ylabel(r'$y$', fontsize=14)
ax2.set_xlabel(r'$x$', fontsize=14)
# ax2.set_xticks(X_test.squeeze())
plt.tight_layout()
plt.show()
