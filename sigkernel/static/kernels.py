from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from sklearn.base import BaseEstimator

from .. import utils

from typing import Optional


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kernel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Kernels.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def fit(X : torch.Tensor, y : Optional[torch.Tensor] = None) -> Kernel:
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None, diag : bool = False) -> torch.Tensor:
        if diag:
            K = self._Kdiag(X)
        else:
            K =  self._K(X, Y)
        return K

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LinearKernel(Kernel):
    """Class for linear (static) kernel."""

    def __init__(self):
        self.static_kernel_type = 'linear'

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        return utils.matrix_mult(X, Y, transpose_Y=True)

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        return utils.squared_norm(X, axis=-1)

class PolynomialKernel(Kernel):
    """Class for polynomial (static) kernel."""

    def __init__(self, degree : float = 3.0, gamma : float = 1.0) -> None:
        self.degree = utils.check_positive_value(degree, 'degree')
        self.gamma = gamma

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.pow(utils.matrix_mult(X, Y, transpose_Y=True) + self.gamma, self.degree)

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        return torch.pow(utils.squared_norm(X, axis=-1) + self.gamma, self.degree)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class StationaryKernel(Kernel):
    """Base class for stationary (static) kernels.

    Warning: This class should not be used directly.
    Use derived classes instead."""

    def __init__(self, sigma : float = 1.0) -> None:
        self.sigma = utils.check_positive_value(sigma, 'sigma')

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        return torch.full((X.shape[0],), self.sigma**2)

class RBFKernel(StationaryKernel):
    """Radial Basis Function aka Gauss (static) kernel ."""

    def __init__(self, sigma : float = 1.0) -> None:
        super().__init__(sigma=sigma)
        self.static_kernel_type = 'rbf'

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = utils.squared_euclid_dist(X, Y) / self.sigma**2
        return torch.exp(-D2_scaled)

class Matern12Kernel(StationaryKernel):
    """Matern12 (static) kernel ."""

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D_scaled = utils.euclid_dist(X, Y) / self.sigma
        return torch.exp(-D_scaled)

class Matern32Kernel(StationaryKernel):
    """Matern32 (static) kernel ."""

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        sqrt3 = torch.sqrt(3.)
        D_scaled = sqrt3 * utils.euclid_dist(X, Y) / self.sigma
        return (1. + D_scaled) * torch.exp(-D_scaled)

class Matern52Kernel(StationaryKernel):
    """Matern52 (static) kernel ."""

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = 5 * utils.squared_euclid_dist(X, Y) / self.sigma**2
        D_scaled = utils.robust_sqrt(D2_scaled)
        return (1. + D_scaled + D2_scaled / 3.) * torch.exp(-D_scaled)

class RationalQuadraticKernel(StationaryKernel):
    """Rational Quadratic (static) kernel ."""

    def __init__(self, sigma : float = 1.0, alpha : float = 1.0) -> None:
        super().__init__(sigma=sigma)
        self.static_kernel_type = 'rq'
        self.alpha = utils.check_positive_value(alpha, 'alpha')

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = utils.squared_euclid_dist(X, Y) / (2 * self.alpha * self.sigma**2)
        return torch.pow((1 + D2_scaled), -self.alpha)

class RBFKernelMix(Kernel):
    """Mixture of RBF kernels"""

    def __init__(self, sigma : list = [0.1, 0.5, 1.0]) -> None:
        self.static_kernel_type = 'rbfmix'
        self.sigma = torch.tensor(sigma).reshape(1,1,-1)

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = utils.squared_euclid_dist(X, Y).unsqueeze(-1) / self.sigma**2
        # print(self.sigma.shape, utils.squared_euclid_dist(X, Y).unsqueeze(-1).shape, D2_scaled.shape)
        return torch.sum(torch.exp(-D2_scaled), dim=-1)

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented.')

class RationalQuadrticKernelMix(Kernel):
    """Mixture of Rational Quadratic kernels"""

    def __init__(self, sigma : float = 1.0, alpha : list = [0.2,0.5,1.,2.,5.]) -> None:
        self.static_kernel_type = 'rqmix'
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.alpha = torch.tensor(alpha)

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = utils.squared_euclid_dist(X, Y).unsqueeze(-1) / (2 * self.alpha.reshape(1,1,-1).to(X.device) * self.sigma**2)
        return torch.pow((1 + D2_scaled), -self.alpha.to(X.device)).sum(dim=-1)

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented.')

class RQMixLinear(Kernel):
    """Mixture of Rational Quadratic and Linear kernels"""

    def __init__(self, sigma : float = 1.0, alpha : list = [0.2,0.5,1.,2.,5.]) -> None:
        self.static_kernel_type = 'rqlinear'
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.alpha = torch.tensor(alpha).unsqueeze(0).unsqueeze(0)

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = utils.squared_euclid_dist(X, Y).unsqueeze(-1) / (2 * self.alpha.reshape(1,1,-1).to(X.device) * self.sigma**2)
        return (torch.pow((1 + D2_scaled), -self.alpha.to(X.device)).sum(dim=-1) + utils.matrix_mult(X, Y, transpose_Y=True))

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented.')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------