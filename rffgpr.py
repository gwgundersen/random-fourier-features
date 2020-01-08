"""============================================================================
Gaussian process regression using random Fourier features. Based on "Random 
Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import numpy as np
from   scipy.spatial.distance import pdist, cdist, squareform
from   scipy.linalg import cholesky, cho_solve


# ------------------------------------------------------------------------------

class RFFGaussianProcessRegressor:

    def __init__(self, rff_dim=10, sigma=1.0):
        """Gaussian process regression using random Fourier features.

        rff_dim : Dimension of random feature.
        sigma :   sigma^2 is the variance.
        """
        self.rff_dim = rff_dim
        self.sigma   = sigma
        self.alpha_  = None
        self.b_      = None
        self.W_      = None

    def fit(self, X, y):
        """Fit model with training data X and target y.
        """
        # NB: We could find alpha using a linear model. However, we could not
        #     compute the covariance matrix in that case.

        # Build kernel approximation using RFFs.
        N, _    = X.shape
        Z, W, b = self._get_rffs(X, return_vars=True)
        sigma_I = self.sigma * np.eye(N)
        self.kernel_ = Z.T @ Z + sigma_I

        # Solve for Rasmussen and William's alpha.
        lower = True
        L = cholesky(self.kernel_, lower=lower)
        self.alpha_ = cho_solve((L, lower), y)

        # Save for `predict` function.
        self.Z_train_ = Z
        self.L_ = L
        self.b_ = b
        self.W_ = W

        return self

    def predict(self, X):
        """Predict using fitted model and testing data X.
        """
        if self.alpha_ is None or self.b_ is None or self.W_ is None:
            msg = "This instance is not fitted yet. Call 'fit' with "\
                  "appropriate arguments before using this method."
            raise NotFit

        Z_test = self._get_rffs(X, return_vars=False)
        K_star = Z_test.T @ self.Z_train_
        y_mean = K_star.dot(self.alpha_)

        lower = True
        v = cho_solve((self.L_, lower), K_star.T)
        y_cov = (Z_test.T @ Z_test) - K_star.dot(v)

        return y_mean, y_cov

    def _get_rffs(self, X, return_vars):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        W, b = self._get_rvs(D)
        B    = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1./ np.sqrt(self.rff_dim)
        Z    = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)
        if return_vars:
            return Z, W, b
        return Z

    def _get_rvs(self, D):
        """On first call, return random variables W and b. Else, return cached
        values.
        """
        if self.W_ is not None:
            return self.W_, self.b_
        W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
        b = np.random.uniform(0, 2*np.pi, size=self.rff_dim)
        return W, b
