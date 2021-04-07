import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GP_Learner:
    def __init__(self, noise_std):
        self.theta = 1.0
        self.length_scale = 1.0
        self.kernel = C(self.theta, (1e-3, 1e3)) * RBF(self.length_scale, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_std**2, normalize_y=False,
                                           n_restarts_optimizer=10)
        self.x_pred = None
        self.y_pred = None
        self.sigma = None

    def learn(self, x, y, x_axis):
        self.gp.fit(x, y)

        self.x_pred = np.atleast_2d(x_axis).T
        self.y_pred, self.sigma = self.gp.predict(self.x_pred, return_std=True)
