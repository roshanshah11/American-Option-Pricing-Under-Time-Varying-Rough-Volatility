#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:05:48 2024

This modul contains a class for computing linear and log signatures for augmented paths, 
for various choices of signature lifts, and polynomial features.

@author: lucapelizzari
"""
import math
import numpy as np
import scipy.special as sc
try:
    import iisignature as ii
except ImportError:
    raise ImportError("The 'iisignature' library is required but not installed. Please install it using 'pip install iisignature'.")

class SignatureComputer:
    """
    Computes signatures of augmented paths
    Attributes:
           T (float): The time horizon for the option in days.
           N (int): The number of time steps.
           K (int): The truncation level for signatures.
           signature_spec (str): Specifies 'linear' or 'log' signatures.
           signature_lift (str): The type of signature lift to apply.
           poly_degree (int): The degree of polynomial features to add.
           days_per_year (int): Number of trading days per year.
    """
    def __init__(self, T, N, K, signature_spec, signature_lift, poly_degree, days_per_year=252):
        """
        Initialize the SignatureComputer.
        
        Args:
            T (float): The time horizon for the option in days.
            N (int): The number of time steps.
            K (int): The truncation level for signatures.
            signature_spec (str): Specifies 'linear' or 'log' signatures.
            signature_lift (str): The type of signature lift to apply.
            poly_degree (int): The degree of polynomial features to add.
            days_per_year (int): Number of trading days per year. Default is 252.
        """
        self.T = T  # Keep original T in days
        self.T_years = T / days_per_year  # Convert to years for financial calculations
        self.N = N
        self.K = K
        self.signature_spec = signature_spec
        self.signature_lift = signature_lift
        self.poly_degree = poly_degree
        self.days_per_year = days_per_year
        
        # Time grid should match what's used in the notebook (scale of T)
        self.tt = np.linspace(0, T, N+1)
        
    def compute_signature(self, X, vol, A, Payoff, dW, I, MM):
        """
        Computes the signature of the augmented path X, vol, A, Payoff
        X = state process, array of Mx(N+1)
        vol = volatility process, array of Mx(N+1)
        A = monoton compononent for augmentation (e.g. time, QV), array of Mx(N+1)
        Payoff = payoff process, array of Mx(N+1)
        dW = Brownian noise, optional
        I = \int vdW
        MM = B

        Output is linear or log signature of augmented path, array of Mx(N+1)x(K+1), 
        with potentially additional polynomial features.
        """
        print(f"Computing {self.signature_spec} signature with {self.signature_lift} lift")
        if self.signature_spec == "linear":
            result = self._compute_linear_signature(X, vol, A, Payoff, dW, I, MM)
        elif self.signature_spec == "log":
            result = self._compute_log_signature(X, vol, A, Payoff, dW, I, MM)
        else:
            raise ValueError(f"Invalid signature_spec: {self.signature_spec}")
        
        return result

    # The rest of the methods remain unchanged
    def _compute_linear_signature(self, X, vol, A, Payoff, dW, I, MM):
        """
        Computes the linear signature of the augmented path X, vol, A, Payoff, for differnet choices of signature lift.
        """
        # Get the actual number of time steps from the data
        actual_N = X.shape[1] - 1
        
        dX = X[:, 1:] - X[:, :-1]
        dvol = vol[:,1:]-vol[:,:-1]
        dI = I[:,1:]-I[:,:-1]
        dM = MM[:,1:]-MM[:,:-1]
        dP = Payoff[:, 1:] - Payoff[:, :-1]
        
        if self.signature_lift == "normal":
            return self._signatureQV(self.tt[:actual_N+1], dX, A[:,:actual_N+1])
        if self.signature_lift == "Brownian":
            return self._signatureQV(self.tt[:actual_N+1], dW[:,:actual_N], A[:,:actual_N+1])
        elif self.signature_lift == "payoff-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            dXX = np.stack((dP, dX), axis=-1)
            return self._signatureQV(self.tt[:actual_N+1], dXX, A[:,:actual_N+1])
        elif self.signature_lift == "delay":
            dW_delay = np.zeros_like(dX)
            dW_delay[:, 1:] = dW[:, :-1]
            dXX = np.stack((dW[:,:actual_N], dW_delay[:,:actual_N]), axis=-1)
            States = np.zeros((X.shape[0],X.shape[1],3))
            States[:,:,2] = vol
            States[:,:,0] = Payoff
            States[:,:,1] = X
            Sig = self._signatureQV(self.tt[:actual_N+1], dXX, A[:,:actual_N+1])
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "polynomial-extended":
            Sig = self._signatureQV(self.tt[:actual_N+1], dX, A[:,:actual_N+1])
            States = np.zeros((X.shape[0],X.shape[1],1))
            States[:,:,0] = Payoff
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "payoff-and-polynomial-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            dXX = np.stack((dX, dP), axis=-1)
            Sig = self._signatureQV(self.tt[:actual_N+1], dXX, A[:,:actual_N+1])
            Poly = self._compute_polynomials_2dim(X,vol)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "logprice-payoff-vol-sig":
            Sig = self._signatureQV(self.tt[:actual_N+1], dvol, A[:,:actual_N+1])
            States = np.zeros((X.shape[0],X.shape[1],3))
            States[:,:,0] = X
            States[:,:,1] = vol
            States[:,:,2] = X*vol
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "vol-payoff-logprice-sig":
            Sig = self._signatureQV(self.tt[:actual_N+1], dX, A[:,:actual_N+1])
            States = np.zeros((X.shape[0],X.shape[1],2))
            States[:,:,0] = vol
            States[:,:,1] = vol*X
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "logprice-vol-Brownian-sig":
            Sig = self._signatureQV(self.tt[:actual_N+1], dW[:,:actual_N], A[:,:actual_N+1])
            States = np.zeros((X.shape[0],X.shape[1],3))
            States[:,:,2] = vol
            States[:,:,0] = Payoff
            States[:,:,1] = X
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "vol-payoff-logprice-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            dXX = np.stack((dP, dW[:,:actual_N]), axis=-1)
            States = np.zeros((X.shape[0],X.shape[1],3))
            States[:,:,0] = Payoff
            States[:,:,1] = X
            States[:,:,2] = vol
            Sig = self._signatureQV(self.tt[:actual_N+1], dXX, A[:,:actual_N+1])
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "price-brownian-lift":
            dXX = np.stack((dX, dW[:,:actual_N]), axis=-1)
            return self._signatureQV(self.tt[:actual_N+1], dXX, A[:,:actual_N+1])
        elif self.signature_lift == "polynomial-vol":
            # Don't use reshape, pass dvol directly
            Sig = self._signatureQV(self.tt[:actual_N+1], dvol[:,:,np.newaxis], A[:,:actual_N+1])
            Poly = self._compute_polynomials(X)
            return np.concatenate((Sig, Poly), axis=-1)
        else:
            raise ValueError(f"Invalid signature_lift for linear signature: {self.signature_lift}")
    
    def _compute_log_signature(self, X, vol, A, Payoff, dW, I, MM):
        """
        Computes the log signature of the augmented path X, vol, A, Payoff, for differnet choices of signature lift.
        """
        # Get actual dimensions from data
        M, T_steps_X = X.shape
        _, T_steps_vol = vol.shape
        _, T_steps_A = A.shape
        
        # Use the minimum dimension for all operations
        common_steps = min(T_steps_X, T_steps_vol, T_steps_A)
        
        # Print dimensions for debugging
        print(f"X shape: {X.shape}, vol shape: {vol.shape}, A shape: {A.shape}")
        print(f"Using {common_steps} time steps for log signature computation")
        
        # Calculate increments based on common dimensions
        dX = X[:, 1:common_steps] - X[:, :common_steps-1]
        dvol = vol[:, 1:common_steps] - vol[:, :common_steps-1]
        
        # Truncate arrays to common dimension
        X_common = X[:, :common_steps]
        vol_common = vol[:, :common_steps]
        A_common = A[:, :common_steps]
        Payoff_common = Payoff[:, :common_steps]
        
        # Calculate Brownian path if needed (using common dimensions)
        if dW is not None and dW.shape[1] >= common_steps-1:
            W = np.zeros((X.shape[0], common_steps))
            W[:,1:] = np.cumsum(dW[:,:common_steps-1], axis=1)
        else:
            W = np.zeros((X.shape[0], common_steps))
        
        if self.signature_lift == "normal":
            XX = np.stack([A_common, X_common], axis=-1)
            return self._full_log_signature(XX)
        elif self.signature_lift == "payoff-extended":
            XX = np.stack([A_common, X_common], axis=-1)
            Sig = self._full_log_signature(XX)
            States = np.zeros((M, common_steps, 1))
            States[:,:,0] = Payoff_common
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "delay":
            X_delay = np.zeros_like(X_common)
            X_delay[:, 1:] = X_common[:, :-1]
            XX = np.stack([A_common, X_common, X_delay], axis=-1)
            return self._full_log_signature(XX)
        elif self.signature_lift == "polynomial-extended":
            XX = np.stack([A_common, vol_common], axis=-1)
            Sig = self._full_log_signature(XX)
            Poly = self._compute_polynomials(X_common)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "payoff-and-polynomial-extended":
            XX = np.stack([A_common, X_common, Payoff_common], axis=-1)
            Sig = self._full_log_signature(XX)
            Poly = self._compute_polynomials_2dim(X_common, vol_common)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "logprice-payoff-vol-sig":
            XX = np.stack([A_common, vol_common], axis=-1)
            Sig = self._full_log_signature(XX)
            States = np.zeros((M, common_steps, 2))
            States[:,:,1] = W
            States[:,:,0] = X_common
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "vol-payoff-logprice-sig":
            XX = np.stack([A_common, X_common], axis=-1)
            Sig = self._full_log_signature(XX)
            States = np.zeros((M, common_steps, 2))
            States[:,:,0] = vol_common
            States[:,:,1] = Payoff_common
            return np.concatenate((Sig, States), axis=-1)
        elif self.signature_lift == "polynomial-vol":
            # Make sure A and vol have the same shape before stacking
            XX = np.stack([A_common, vol_common], axis=-1)
            Sig = self._full_log_signature_dim_two_level_three(XX, self.K)
            Poly = self._compute_polynomials(X_common)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "logprice-vol-Brownian-sig":
            XX = np.stack([A_common, W], axis=-1)
            Sig = self._full_log_signature(XX)
            States = np.zeros((M, common_steps, 3))
            States[:,:,2] = vol_common*X_common
            States[:,:,0] = X_common
            States[:,:,1] = vol_common
            return np.concatenate((Sig, States), axis=-1)
        else:
            raise ValueError(f"Invalid signature_lift for log signature: {self.signature_lift}")

    def _compute_polynomials(self, X):
        """
        Computes the Laguerre polynomials of X
        """
        Polynomials = np.zeros((X.shape[0], X.shape[1], self.poly_degree))
        for k in range(self.poly_degree):
            Polynomials[:,:,k] = sc.laguerre(k+1)(X)
        return Polynomials
    
    def _compute_polynomials_2dim(self, X, vol):
        """
        Computes the Laguerre polynomials of (X,vol)
        """
        DD_primal = int((self.poly_degree+1)*(self.poly_degree+2)/2) #Number of polynomials 2 dim
        Polynomials = np.zeros((X.shape[0], X.shape[1], DD_primal))
        for k in range(self.poly_degree+1):
            for j in range(0,k+1):
                C = np.zeros((self.poly_degree+1,self.poly_degree+1))
                C[k,j] = 1
                Polynomials[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X,vol, C)
        return Polynomials
    
    def _signature_ONB_basis(self, tGrid, X, deg):
        sig = np.zeros((X.shape[0],X.shape[1],deg))
        for i in range(deg):
            sig[:,1:,i]= 1/math.factorial(i+1)*np.cumsum(X[:,:-1]*(tGrid[:-1])**(i+1),axis=1)/(X.shape[1])
        return sig
    
    def _signatureQV(self, tGrid, dx, QV):
        """
        Compute the signature of a path (t,x,[x]) up to degree K.

        Parameters
        ----------
        tGrid : numpy array
            Time grid of t, size N+1.
        dx : numpy array
            Increments of the path x, an array of dimension MxNxd.
        QV : numpy array
            Quadratic variation of the path.

        Returns
        -------
        sig : numpy array
            The signature of (t,x,[x]) at all the times, an array of size (M,N,k+1).
        """
        M, d, z = self._prepare_sigQV(tGrid, dx, QV)
        
        # We need to compute the signature of z
        k = ii.siglength(d+1, self.K)
        N = len(tGrid)
        
        sig = np.zeros((M, N, k+1))
        for m in range(M):
            sig[m, 1:N, 1:k+1] = ii.sig(z[m, :, :], self.K, 2)
        sig[:, :, 0] = 1

        return sig
    
    def _prepare_sigQV(self, tGrid, dx, QV):
        """Auxiliary function for computing signatures. See help for signature."""
        N = len(tGrid) - 1
        if len(dx.shape) == 1:
            # assume that M = d = 1
            dx = dx.reshape((1, dx.shape[0], 1))
        if len(dx.shape) == 2:
            # assume that either d = 1 or M = 1
            if dx.shape[0] == N:
                dx = dx.reshape((1, dx.shape[0], dx.shape[1]))
            elif dx.shape[1] == N:
                dx = dx.reshape((dx.shape[0], dx.shape[1], 1))
        assert len(dx.shape) == 3 and dx.shape[1] == N, \
            f"dx is misshaped as {dx.shape}"
        M, _, d = dx.shape
        QV = QV.reshape(M, N+1, 1)
        x = np.zeros((M, N+1, d))
        x[:, 1:(N+1), :] = np.cumsum(dx, axis=1)
        z = np.concatenate((QV, x), axis=2)
        return M, d, z  # d+1 because we added the QV dimension

    def _full_log_signature(self, X):
        """
        Compute the full log signature of the given paths.

        Args:
            X (np.ndarray): The paths to compute the log signature for.

        Returns:
            np.ndarray: The computed log signatures.
        """
        # Batch computation to avoid buffer overflow on large batch sizes
        m, n, d = X.shape
        L = ii.logsiglength(d, self.K)
        log_sig = np.zeros((m, n, L))
        bch = ii.prepare(d, self.K, 'C')  # precalculate the BCH formula
        batch_size = 2048  # adjust batch size as needed
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X[start:end]  # shape (batch, n, d)
            for i in range(1, n):
                # compute logsig for each time step i for batch
                log_sig[start:end, i] = ii.logsig(X_batch[:, :i+1], bch, 'C')
        return log_sig
    
    def _full_log_signature_dim_two_level_three(self, X, deg):
        """
        Compute the full log signature for 2D paths up to level 3.

        Args:
            X (np.ndarray): The paths to compute the log signature for.
            deg (int): The degree of the log signature (1, 2, or 3).

        Returns:
            np.ndarray: The computed log signatures.

        Raises:
            AssertionError: If the input dimensions are incorrect or deg is out of range.
        """
        m, n, d = X.shape
        
        assert d == 2
        assert (deg >= 1) and (deg <= 3)
        
        log_sig_dim = {1: 2, 2: 3, 3: 5}
        
        log_sig = np.zeros((m, n, log_sig_dim[deg]))
        
        log_sig[:,1:,:2] = X[:, 1:] - X[:, 0].reshape(-1, 1, 2)
            
        if deg >= 2:
            dX = np.diff(X, axis=1)
            
            for i in range(1, n):
                l = log_sig[:, i - 1]
                dx = dX[:, i - 1]
                log_sig[:, i, 2] = l[:, 2] + 0.5 * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                
        if deg == 3:
            for i in range(1, n):
                l = log_sig[:, i - 1]
                dx = dX[:, i - 1]
                log_sig[:, i, 3] = l[:, 3] - 0.5 * l[:, 2] * dx[:, 0] + (1 / 12) * \
                        (l[:, 0] - dx[:, 0]) * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                log_sig[:, i, 4] = l[:, 4] + 0.5 * l[:, 2] * dx[:, 1] - (1 / 12) * \
                        (l[:, 1] - dx[:, 1]) * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                
        return log_sig
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
