import numpy as np
from utils import *

class DynamicHurstrBergomi(object):
    """
    Modified rBergomi class that supports time-varying Hurst parameters.
    """
    def __init__(self, n=100, N=1000, T=1.00, H_series=None):
        """
        Constructor for class with dynamic Hurst parameter.
        
        Parameters:
        -----------
        n : int
            Granularity (steps per year)
        N : int
            Number of paths
        T : float
            Maturity
        H_series : array-like
            Time series of Hurst parameters for each time step
        """
        # Basic assignments
        self.T = T  # Maturity
        self.n = n  # Granularity (steps per year)
        self.dt = 1.0/self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :]  # Time grid
        self.N = N  # Paths
        
        # Handle Hurst parameter series
        if H_series is None:
            self.H_series = np.full(1 + self.s, 0.07)  # Default value if not provided
        else:
            # Interpolate if lengths don't match
            if len(H_series) != 1 + self.s:
                t_input = np.linspace(0, self.T, len(H_series))
                t_output = np.linspace(0, self.T, 1 + self.s)
                self.H_series = np.interp(t_output, t_input, H_series)
            else:
                self.H_series = H_series
        
        # Convert H to alpha series (a = H - 0.5)
        self.a_series = self.H_series - 0.5

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure, adapting to time-varying H.
        """
        # We need to generate correlated Brownian increments for each time step
        # with correlation structure depending on the local Hurst parameter
        
        dW = np.zeros((self.N, self.s, 2))
        
        # Generate standard normal random variables
        Z1 = np.random.randn(self.N, self.s)
        Z2 = np.random.randn(self.N, self.s)
        
        # For each time step, apply the appropriate correlation structure
        for i in range(self.s):
            a_i = self.a_series[i]
            c_i = cov(a_i, self.n)  # Get covariance for this specific Hurst value
            
            # Apply correlation structure
            dW[:, i, 0] = Z1[:, i]
            dW[:, i, 1] = c_i[0, 1] * Z1[:, i] + np.sqrt(1 - c_i[0, 1]**2) * Z2[:, i]
            
        return dW * np.sqrt(self.dt)

    def Y(self, dW):
        """
        Constructs Volterra process with time-varying Hurst parameter.
        """
        Y = np.zeros((self.N, 1 + self.s))
        
        # For each time step, compute Y with the corresponding Hurst parameter
        for t in range(1, 1 + self.s):
            # Get current alpha value
            a_t = self.a_series[t-1]
            
            # Apply the Volterra kernel with current Hurst parameter
            if a_t == 0:
                # For H = 0.5 (a = 0), Y is simply the Brownian motion W^1
                Y[:, 1:t+1] = np.cumsum(dW[:, :t, 0], axis=1)
            else:
                # For a time-varying kernel, we need to use the appropriate
                # weights for each time step
                for i in range(1, t+1):
                    kernel_weight = (t - i + 1)**(a_t) - (t - i)**(a_t)
                    if i == 1:
                        Y[:, t] += dW[:, i-1, 0] * kernel_weight
                    else:
                        Y[:, t] += dW[:, i-1, 0] * kernel_weight
                
                # Scale by the appropriate factor
                Y[:, t] *= np.sqrt(2 * a_t + 1)
        
        return Y

    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi=1.0, eta=1.0):
        """
        rBergomi variance process with time-varying Hurst parameter.
        """
        self.xi = xi
        self.eta = eta
        
        V = np.zeros((self.N, 1 + self.s))
        V[:, 0] = xi  # Initial value
        
        for t in range(1, 1 + self.s):
            a_t = self.a_series[t]
            t_val = self.t[0, t]
            
            # Apply the volatility formula with current Hurst parameter
            V[:, t] = xi * np.exp(eta * Y[:, t] - 0.5 * eta**2 * t_val**(2 * a_t + 1))
        
        return V

    def S(self, V, dB, S0=1):
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

def SimulationWithDynamicHurst(M, N, T_years, phi, rho, K, X0, H_series, xi, eta, r, days_per_year=252):
    """
    Simulate paths of rBergomi price, volatility, Brownian motions and I with time-varying Hurst parameter.
    
    Parameters:
    -----------
    M : int
        Number of samples in first simulation, used for LS-regression
    N : int
        Number of discretization points for the grid [0,T]
    T_years : float
        Maturity in years
    phi : function
        Payoff function (Put or Call)
    rho : float
        Correlation coefficient
    K : int
        Depth of Signature
    X0 : float
        Initial value of log-price (log(S0))
    H_series : array-like
        Time series of Hurst parameters for each time step
    xi, eta : float
        Specifications for rBergomi volatility process
    r : float
        Interest-rate
    days_per_year : int
        Number of days in a year (default = 252)
    """
    
    # Create the dynamic Hurst rBergomi model
    rB = DynamicHurstrBergomi(N, M, T_years, H_series)
    
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
        I[:, n+1] = I[:, n] + np.sqrt(V[:, n]) * dW1[:, n, 0]
    
    dI = I[:, 1:steps] - I[:, 0:time_steps]
    dI = dI.reshape(M, time_steps, 1)
    
    return X, V, I, dI, dW1, dW2, dB, Y

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a Dynamic Hurst-Heston stochastic volatility model
that supports a time‐varying Hurst exponent H(t) in the diffusion term.
Designed to plug into your signature‐based American option pricer
(similar interface to DynamicHurstrBergomi).
"""

class DynamicHurstHeston:
    """
    Dynamic Hurst‐Heston model:
        dv_t = κ (θ − v_t) dt + ξ * v_t^{H(t)} dW^v_t
        dS_t = μ S_t dt    + sqrt(v_t)    S_t dW^S_t
    where H(t) > 0.5 is provided as a time series.
    """

    def __init__(self,
                 n: int = 100,
                 N: int = 1000,
                 T: float = 1.0,
                 H_series: np.ndarray = None):
        """
        Args:
            n         -- steps per year
            N         -- number of Monte-Carlo paths
            T         -- maturity (years)
            H_series  -- length (n*T+1) array of Hurst exponents, each > 0.5
        """
        self.T = T
        self.n = n
        self.dt = 1.0 / n
        self.s = int(n * T)
        self.t = np.linspace(0, T, self.s + 1)
        self.N = N

        # build H_series, interpolate if needed
        if H_series is None:
            self.H = np.full(self.s + 1, 0.7)
        else:
            x = np.linspace(0, T, len(H_series))
            y = np.array(H_series)
            self.H = np.interp(self.t, x, y)
        # ensure H>0.5
        self.H = np.maximum(self.H, 0.5001)

    def dW1(self) -> np.ndarray:
        """
        Volatility Brownian increments for v_t
        (could be fractional—but here we use standard BM)
        Returns shape (N, s)
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dW2(self) -> np.ndarray:
        """
        Orthogonal increments for S_t
        Returns shape (N, s)
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self,
           dW1: np.ndarray,
           dW2: np.ndarray,
           rho: float = 0.0) -> np.ndarray:
        """
        Correlated price increments: dB = rho dW1 + sqrt(1−rho²) dW2
        """
        return rho * dW1 + np.sqrt(1 - rho ** 2) * dW2

    def V(self,
          dW1: np.ndarray,
          kappa: float = 1.0,
          theta: float = 0.04,
          xi: float = 0.5,
          v0: float = None) -> np.ndarray:
        """
        Simulate variance paths under Dynamic Hurst‐Heston.
        Returns V array of shape (N, s+1).
        """
        if v0 is None:
            v0 = theta
        V = np.zeros((self.N, self.s + 1))
        V[:, 0] = v0

        for t in range(self.s):
            H_t = self.H[t]
            drift = kappa * (theta - V[:, t]) * self.dt
            diffusion = xi * (V[:, t] ** H_t) * dW1[:, t]
            V[:, t + 1] = np.maximum(0.0, V[:, t] + drift + diffusion)
        return V

    def S(self,
          V: np.ndarray,
          dB: np.ndarray,
          mu: float = 0.0,
          S0: float = 1.0) -> np.ndarray:
        """
        Simulate asset price paths under Dynamic Hurst‐Heston.
        Returns S array of shape (N, s+1).
        """
        S = np.zeros_like(V)
        S[:, 0] = S0

        # Euler–Maruyama on log scale
        for t in range(self.s):
            S[:, t + 1] = S[:, t] * np.exp(
                (mu - 0.5 * V[:, t]) * self.dt
                + np.sqrt(V[:, t]) * dB[:, t]
            )
        return S

def SimulationWithDynamicHurstHeston(M, N, T_years, phi, rho, K, X0, H_series, xi, eta, r, days_per_year=252):
    """
    Simulate paths under Dynamic Hurst-Heston model with an interface matching SimulationWithDynamicHurst.
    
    Parameters:
    -----------
    M : int
        Number of samples in simulation
    N : int
        Number of discretization points for the grid [0,T]
    T_years : float
        Maturity in years
    phi : function
        Payoff function (unused in Heston, included for interface compatibility)
    rho : float
        Correlation coefficient
    K : int
        Depth of Signature (unused in Heston, included for interface compatibility)
    X0 : float
        Initial value of price
    H_series : array-like
        Time series of Hurst parameters for each time step
    xi : float
        Vol-of-vol parameter in the Heston model
    eta : float
        Unused in Heston but kept for interface compatibility
    r : float
        Risk-free interest rate
    days_per_year : int
        Number of days in a year (default = 252, unused in Heston but kept for compatibility)
    
    Returns:
    --------
    Tuple containing:
        X   -- price paths (M, s+1)
        V   -- variance paths (M, s+1)
        I   -- integrated variance (M, s+1)
        dI  -- variance increments (M, s, 1)
        dW1 -- volatility BM increments (M, s, 1)
        dW2 -- orthogonal BM increments (M, s, 1)
        dB  -- correlated price increments (M, s, 1)
        Y   -- Yamada process (placeholder, unused in Heston)
    """
    # The Heston-specific parameters with default values
    kappa = 2.0  # Mean reversion speed
    theta = 0.04  # Long-term variance
    mu = r  # Risk-free rate (used in price process)
    
    # Create the dynamic Hurst-Heston model
    model = DynamicHurstHeston(n=N, N=M, T=T_years, H_series=H_series)
    
    # Two independent Brownian motion increments
    dW1 = model.dW1()
    dW2 = model.dW2()
    
    # Variance process V, using Heston model's implementation
    V = model.V(dW1, kappa, theta, xi)
    
    # Price process with correlated increments
    dB = model.dB(dW1, dW2, rho)
    X = model.S(V, dB, mu + r, X0)
    
    # Get the steps for consistent indexing
    steps = model.s + 1
    time_steps = model.s
    
    # Calculate integrated variance (I) and its increments (dI)
    I = np.zeros((M, steps))
    for t in range(time_steps):
        I[:, t+1] = I[:, t] + np.sqrt(V[:, t]) * dW1[:, t]
    
    dI = (I[:, 1:steps] - I[:, 0:time_steps]).reshape(M, time_steps, 1)
    
    # Reshape increments for signature compatibility
    dW1_reshaped = dW1.reshape(M, time_steps, 1)
    dW2_reshaped = dW2.reshape(M, time_steps, 1)
    dB_reshaped = dB.reshape(M, time_steps, 1)
    
    # For interface compatibility with rBergomi, create a placeholder Y process
    Y = np.zeros_like(V)
    
    return X, V, I, dI, dW1_reshaped, dW2_reshaped, dB_reshaped, Y