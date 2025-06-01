#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation implementation of the rough Bergomi (rBergomi) model.

This module provides tools for simulating price paths under the rBergomi model,
which is described in "Pricing under rough volatility". The implementation is
inspired by https://github.com/ryanmccrickerd/rough_bergomi

References:
    Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
    Quantitative Finance, 16(6), 887-904.
"""
from typing import Tuple
import numpy as np
from FBM_package import FBM
from utils import *

class rBergomi:
    """Implementation of the rough Bergomi (rBergomi) model simulation."""
    
    def __init__(self, n: int = 100, N: int = 1000, T: float = 1.00, a: float = -0.4) -> None:
        """
        Initialize the rBergomi model.

        Args:
            n: Number of steps per year (granularity)
            N: Number of paths to simulate
            T: Maturity in years
            a: Alpha parameter (related to Hurst parameter by H = a + 0.5)
        """
        self.T = T
        self.n = n
        self.dt = 1.0/self.n
        self.s = int(self.n * self.T)
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:]
        self.a = a
        self.N = N
        
        # Correlation structure for hybrid scheme (kappa = 1)
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self) -> np.ndarray:
        """
        Produces random numbers for variance process with required
        covariance structure.

        Returns:
            Random numbers with required covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW: np.ndarray) -> np.ndarray:
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments. If a = 0, Y is simply the Brownian motion W^1.

        Args:
            dW: Correlated 2d Brownian increments.

        Returns:
            Volterra process.
        """
        if self.a == 0:
            Y = np.zeros((self.N, 1 + self.n))
            Y[:,1:1+self.n] = np.cumsum(dW[:,:,0], axis = 1)
            return Y
        else:
            Y1 = np.zeros((self.N, 1 + self.s)) # Exact integrals
            Y2 = np.zeros((self.N, 1 + self.s)) # Riemann sums

            # Construct Y1 through exact integral
            for i in np.arange(1, 1 + self.s, 1):
                Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

            # Construct arrays for convolution
            G = np.zeros(1 + self.s) # Gamma
            for k in np.arange(2, 1 + self.s, 1):
                G[k] = g(b(k, self.a)/self.n, self.a)

            X = dW[:,:,0] # Xi

            # Initialise convolution result, GX
            GX = np.zeros((self.N, len(X[0,:]) + len(G) - 1))

            # Compute convolution, FFT not used for small n
            # Possible to compute for all paths in C-layer?
            for i in range(self.N):
                GX[i,:] = np.convolve(G, X[i,:])

            # Extract appropriate part of convolution
            Y2 = GX[:,:1 + self.s]

            # Finally construct and return full process
            Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
            return Y

    def dW2(self) -> np.ndarray:
        """
        Obtain orthogonal increments.

        Returns:
            Orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1: np.ndarray, dW2: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """
        Constructs correlated price Brownian increments, dB.

        Args:
            dW1: First Brownian increments.
            dW2: Second Brownian increments.
            rho: Correlation coefficient.

        Returns:
            Correlated price Brownian increments.
        """
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y: np.ndarray, xi: float = 1.0, eta: float = 1.0) -> np.ndarray:
        """
        rBergomi variance process.

        Args:
            Y: Volterra process.
            xi: Scale parameter for volatility process.
            eta: Volatility of volatility.

        Returns:
            Variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        return V

    def S(self, V: np.ndarray, dB: np.ndarray, S0: float = 1) -> np.ndarray:
        """
        rBergomi price process.

        Args:
            V: Variance process.
            dB: Correlated price Brownian increments.
            S0: Initial price.

        Returns:
            Price process.
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S

    def S1(self, V: np.ndarray, dW1: np.ndarray, rho: float, S0: float = 1) -> np.ndarray:
        """
        rBergomi parallel price process.

        Args:
            V: Variance process.
            dW1: First Brownian increments.
            rho: Correlation coefficient.
            S0: Initial price.

        Returns:
            Parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S

def SimulationofrBergomi(
    M: int,
    N: int,
    T_years: float,
    phi: callable,
    rho: float,
    K: int,
    X0: float,
    H: float,
    xi: float,
    eta: float,
    r: float,
    days_per_year: int = 252) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate paths of rBergomi model including price, volatility, and Brownian motions.

    Args:
        M: Number of samples for LS-regression
        N: Number of discretization points
        T_years: Maturity in years
        phi: Payoff function (Put or Call)
        rho: Correlation coefficient
        K: Signature depth
        X0: Initial log-price
        H: Hurst parameter
        xi: Scale parameter for volatility process
        eta: Volatility of volatility
        r: Risk-free interest rate
        days_per_year: Trading days per year (default=252)

    Returns:
        Tuple containing:
        - X: Price paths
        - V: Volatility paths
        - I: Integrated variance
        - dI: Variance increments
        - dW1: First Brownian increments
        - dW2: Second Brownian increments
        - dB: Correlated price Brownian increments
        - Y: Volterra process
    """
    # Initialize rBergomi model
    rB = rBergomi(N, M, T_years, H - 0.5)
    
    # Two independent Brownian motion increments
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    
    # Volatility process V, array of Mx(N+1)
    Y = rB.Y(dW1)
    V = rB.V(Y, xi, eta)
    
    # Price-process in rBergomi
    dB = rB.dB(dW1, dW2, rho)
    X = rB.S(V, dB)  # array of Mx(N+1)
    
    # Get the shape of X to determine the time grid
    M_paths, N_times = X.shape
    
    # Create a time grid that matches the shape of X
    tt = np.linspace(0, T_years, N_times)
    
    # Apply the interest rate factor with broadcasting that matches X's shape
    r_factor = np.exp(r * tt)
    X = X0 * X * r_factor[np.newaxis, :]
    
    # Calculate time steps based on actual array shape
    steps = N_times
    time_steps = steps - 1  # Adjust for zero-indexing
    I = np.zeros(shape=(M, steps))
    
    for n in range(time_steps):
        I[:,n+1] = I[:,n] + np.sqrt(V[:,n])*dW1[:,n,0]
    
    dI = I[:,1:steps] - I[:,0:time_steps]
    dI = dI.reshape(M, time_steps, 1)
    
    return X, V, I, dI, dW1, dW2, dB, Y


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Hurst-Heston stochastic volatility model
using the Volterra‐covariance approach instead of FBM.
"""
class HurstHeston:
    """
    Hurst-Heston with constant H:
      dv_t = κ(θ - v_t)dt + ξ * v_t^H * dW^v_t
      dS_t = μ S_t dt    + sqrt(v_t) * S_t * dW^S_t
    where dW^v has the Volterra covariance structure for Hurst H.
    """
    def __init__(self, n=100, N=1000, T=1.0, H=0.7):
        self.T     = T
        self.n     = n
        self.dt    = 1.0/n
        self.s     = int(n*T)
        self.t     = np.linspace(0, T, self.s+1)
        self.N     = N
        self.H     = H
        # precompute covariance matrix for alpha = H-0.5
        self.alpha = H - 0.5
        self.cov   = cov(self.alpha, self.n)

    def dW1(self) -> np.ndarray:
        """
        Volterra‐Gaussian increments for v_t:
        shape = (N, s), each row ~ N(0, cov)
        """
        # draw N independent multivariate normals of dim s
        # using the same covariance at each step
        # but rBergomi did: cov for increments at each time-lag,
        # here we approximate by stationary cov on grid
        # so we simulate (N x s)-paths at once:
        flat = np.random.multivariate_normal(
            mean=np.zeros(self.s),
            cov=self.cov,
            size=self.N
        )
        return flat * np.sqrt(self.dt)

    def dW2(self) -> np.ndarray:
        """Orthogonal Brownian increments for S_t"""
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0) -> np.ndarray:
        """Correlate price increments with volatility increments."""
        return rho * dW1 + np.sqrt(1 - rho**2) * dW2

    def V(self, dW1, kappa=1.0, theta=0.04, xi=0.5, v0=None) -> np.ndarray:
        """Simulate variance under Hurst-Heston."""
        if v0 is None: v0 = theta
        V = np.zeros((self.N, self.s+1))
        V[:,0] = v0
        for t in range(self.s):
            drift     = kappa * (theta - V[:,t]) * self.dt
            diffusion = xi * (V[:,t]**self.H) * dW1[:,t]
            V[:,t+1]  = np.maximum(0.0, V[:,t] + drift + diffusion)
        return V

    def S(self, V, dB, mu=0.0, S0=1.0) -> np.ndarray:
        """Simulate asset prices under Hurst-Heston."""
        S = np.zeros_like(V)
        S[:,0] = S0
        for t in range(self.s):
            S[:,t+1] = S[:,t] * np.exp(
                (mu - 0.5*V[:,t])*self.dt
                + np.sqrt(V[:,t]) * dB[:,t]
            )
        return S

# Wrapper to match your pipeline:
def SimulationofHurstHeston(
    M, n, T, rho, X0, H, kappa, theta, xi, mu=0.0
):
    model = HurstHeston(n=n, N=M, T=T, H=H)
    dW1  = model.dW1()
    dW2  = model.dW2()
    V    = model.V(dW1, kappa, theta, xi)
    dB   = model.dB(dW1, dW2, rho)
    X    = model.S(V, dB, mu, X0)

    # integrated variance
    I = np.zeros_like(V)
    for t in range(model.s):
        I[:,t+1] = I[:,t] + np.sqrt(V[:,t]) * dW1[:,t]
    dI = (I[:,1:] - I[:,:-1]).reshape(M, model.s, 1)

    return X, V, I, dI, dW1.reshape(M, model.s,1), dW2.reshape(M, model.s,1), dB.reshape(M, model.s,1)
