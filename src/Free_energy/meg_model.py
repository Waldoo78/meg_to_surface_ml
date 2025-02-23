import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

class MEGModel:
    def __init__(self, Y, L, n_components=23):
        """
        Initialize MEG model with data reduction.
        
        Parameters
        ----------
        Y : array-like
            MEG measurements (n_channels x n_times)
        L : array-like
            Leadfield matrix (n_channels x n_sources)
        n_components : int
            Number of components for SVD reduction
        """
        self.Nc, self.Nt = Y.shape
        self.Ns = L.shape[1]
        
        # Convert to JAX arrays
        self.Y = jnp.array(Y)
        self.L = jnp.array(L)

        # SVD reduction
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        self.k = min(n_components, len(s))
        self.U_k = U[:, :self.k]

        # Reduce data dimensions
        self.Y_reduced = self.U_k.T @ self.Y
        self.L_reduced = self.U_k.T @ self.L

        # Compute empirical covariance and its inverse
        self.Sigma_Y_emp = (self.Y_reduced @ self.Y_reduced.T) / self.Nt
        self.inv_Y = jnp.linalg.inv(self.Sigma_Y_emp)

        # Regularize M matrix
        M = self.L_reduced.T @ self.inv_Y @ self.L_reduced
        lambda_reg_M = 1e-19 * np.trace(M) / M.shape[0]
        M_reg = M + lambda_reg_M * np.eye(M.shape[0], dtype=np.float64)
        self.Qa = jnp.linalg.inv(M_reg)

        # Prior parameters
        self.nu = jnp.array([-40.0, -40.0])
        self.Pi = 1e-3 * jnp.eye(2)

        # JIT compile methods
        self.build_sigma_theo = jit(self.build_sigma_theo)
        self.compute_free_energy = jit(self.compute_free_energy)
        self._compute_gradient = jit(grad(self.compute_free_energy))

    def build_sigma_theo(self, theta):
        """
        Build theoretical covariance matrix.
        
        Parameters
        ----------
        theta : array-like
            Parameters [log_sigma2, log_gamma]
        
        Returns
        -------
        array-like
            Theoretical covariance matrix
        """
        log_sigma2, log_gamma = theta
        sigma2 = jnp.exp(log_sigma2)
        gamma = jnp.exp(log_gamma)
        return sigma2 * jnp.eye(self.k) + gamma * (self.L_reduced @ self.Qa @ self.L_reduced.T)

    def compute_free_energy(self, theta):
        """
        Compute free energy for given parameters.
        
        Parameters
        ----------
        theta : array-like
            Parameters [log_sigma2, log_gamma]
            
        Returns
        -------
        float
            Free energy value
        """
        Sigma_th = self.build_sigma_theo(theta)
        inv_Sigma_th = jnp.linalg.inv(Sigma_th)

        sign, logdet = jnp.linalg.slogdet(Sigma_th)

        trace_term = -(self.Nt / 2.0) * jnp.trace(self.Sigma_Y_emp @ inv_Sigma_th)
        log_det_term = -(self.Nt / 2.0) * logdet

        diff = theta - self.nu
        prior_term = -0.5 * diff.T @ (self.Pi @ diff)

        const_term = -(self.Nt * self.k / 2.0) * jnp.log(2.0 * jnp.pi)

        return trace_term + log_det_term + prior_term + const_term

    def compute_gradient(self, theta):
        """
        Compute gradient of free energy.
        
        Parameters
        ----------
        theta : array-like
            Parameters [log_sigma2, log_gamma]
            
        Returns
        -------
        array-like
            Gradient of free energy
        """
        return self._compute_gradient(theta)

    def solve(self, theta):
        """
        Compute posterior activations.
        
        Parameters
        ----------
        theta : array-like
            Optimal parameters [log_sigma2, log_gamma]
            
        Returns
        -------
        array-like
            Posterior activations
        """
        Sigma_th = self.build_sigma_theo(theta)
        inv_Sigma_th = jnp.linalg.inv(Sigma_th)

        log_sigma2, log_gamma = theta
        gamma = jnp.exp(log_gamma)

        a_post_reduced = gamma * (self.L_reduced.T @ (inv_Sigma_th @ self.Y_reduced))

        return a_post_reduced