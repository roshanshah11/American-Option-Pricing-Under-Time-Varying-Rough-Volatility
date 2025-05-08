#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:15:45 2024

Simulation of rBergomi model, see "Pricing under rough volatility"
We use the simulation package https://github.com/ryanmccrickerd/rough_bergomi

@author: lucapelizzari
"""
import numpy as np
from FBM_package import FBM
import scipy as sc


"""Manually Putting the rBergomi package b/c stuff does NOT work:import numpy as np"""

from utils import *

class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """
    def __init__(self, n = 100, N = 1000, T = 1.00, a = -0.4):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n # Granularity (steps per year)
        self.dt = 1.0/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:] # Time grid
        self.a = a # Alpha
        self.N = N # Paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments. If a = 0, Y is simply the Brownian motion W^1.
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

            # Init ialise convolution result, GX
            GX = np.zeros((self.N, len(X[0,:]) + len(G) - 1))

            # Compute convolution, FFT not used for small n
            # Possible to compute for all paths in C-layer?
            for i in range(self.N):
                GX[i,:] = np.convolve(G, X[i,:])

            # Extract appropriate part of convolution
            Y2 = GX[:,:1 + self.s]

            # Finally contruct and return full process
            Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
            return Y
    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho = 0.0):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi = 1.0, eta = 1.0):
        """
        rBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        return V

    def S(self, V, dB, S0 = 1):
        """
        rBergomi price process.
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

    def S1(self, V, dW1, rho, S0 = 1):
        """
        rBergomi parallel price process.
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

def SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r, days_per_year=252):
    """Simulate paths of rBergomi price, volatility, Brownian motions and I
    M = Number of samples in first simulation, used for LS-regression
    N = Number of discretization points for the grid [0,T]
    T = maturity in days
    phi = payoff functions (Put or Call)
    rho = correlation coefficient
    K = depth of Signature
    X0 = log(S0) initial value of log-price
    H = Hurst-parameter for fBm in rBergomi model
    xi,eta = specifications for rBergomi volatility process
    r = interest-rate
    days_per_year = number of days in a year (default = 252)
    """
    # Convert T from days to years
    T_years = T / days_per_year
    
    # Using rBergomi-Package for volatility and Brownian motions
    rB = rBergomi(N, M, T_years, -0.5+H)
    
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