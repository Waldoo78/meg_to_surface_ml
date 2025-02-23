import jax
import jax.numpy as jnp
from jax import grad, jit, hessian
from functools import partial
import numpy as np

class MEGOptimizer:
    def __init__(self, Y, L, n_components=32, 
                 starts=None, nu=None, Pi=None, 
                 lambda_reg=1e-12):
        self.Nc, self.Nt = Y.shape
        self.Ns = L.shape[1]
        
        self.Y = jnp.array(Y)
        self.L = jnp.array(L)

        # SVD reduction
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        self.k = min(n_components, len(s))
        self.U_k = U[:, :self.k]

        # Data reduction
        Y_centered = self.Y 
        self.Y_reduced = self.U_k.T @ Y_centered
        self.L_reduced = self.U_k.T @ self.L

        # Empirical covariance
        self.Sigma_Y_emp = (self.Y_reduced @ self.Y_reduced.T) / (self.Nt)
        self.inv_Y = jnp.linalg.inv(self.Sigma_Y_emp)
        
        # Regularization
        M = self.L_reduced.T @ (self.inv_Y @ self.L_reduced)/ (self.Nt)
        lambda_reg_M = lambda_reg * np.trace(M) / M.shape[0] 
        M_reg = M + lambda_reg_M * np.eye(M.shape[0], dtype=np.float64)
        self.Qa = jnp.linalg.inv(M_reg)

        # Prior parameters
        self.nu = jnp.array([-32.0, -32.0]) if nu is None else jnp.array(nu)
        self.Pi = 1/256 * jnp.eye(2) if Pi is None else jnp.array(Pi)

        # Initialize starting points
        if starts is None:
            self.starts = self._create_default_grid_points()
        else:
            self.starts = [jnp.array(point) for point in starts]

        # JIT compilation
        self.build_sigma_theo = jit(self.build_sigma_theo)
        self.compute_free_energy = jit(self.compute_free_energy)
        self._compute_gradient = jit(grad(self.compute_free_energy))

    def _create_default_grid_points(self):
        """Create default grid points if none are provided"""
        log_sigma2_range = jnp.array([-20.0])
        log_gamma_range = jnp.array([-10.0])
        
        grid_points = []
        for log_s2 in log_sigma2_range:
            for log_g in log_gamma_range:
                grid_points.append(jnp.array([float(log_s2), float(log_g)]))
        
        return grid_points

    def build_sigma_theo(self, theta):
        log_sigma2, log_gamma = theta
        sigma2 = jnp.exp(log_sigma2)
        gamma = jnp.exp(log_gamma)
        return sigma2 * jnp.eye(self.k) + gamma * (self.L_reduced @ self.Qa @ self.L_reduced.T)

    def compute_free_energy(self, theta):
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
        return self._compute_gradient(theta)

    def compute_hessian(self, theta):
        """Calcule la Hessienne de la Free Energy"""
        return hessian(self.compute_free_energy)(theta)

    @partial(jit, static_argnums=(0,))
    def _adam_step(self, theta, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        g = self.compute_gradient(theta)
        grad_norm = jnp.linalg.norm(g)
        
        # Adaptive learning rates based on gradient norm
        base_alphas = jnp.array([1e-2, 1e-4])  
        alphas = base_alphas
        
        # More conservative gradient clipping
        max_grad_norm = 1000.0
        g = jnp.where(grad_norm > max_grad_norm, 
                      g * (max_grad_norm / grad_norm), 
                      g)
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        update = alphas * m_hat / (jnp.sqrt(v_hat) + epsilon)
        return theta + update, m, v, g, grad_norm

    def optimize(self, maxiter=1000, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                tol=1e-6, log_every=1, coarse_search=True):
        best_result = None
        best_fe = -float('inf')

        if coarse_search:
            print("\n=== Coarse Grid Search ===")
            initial_energies = []
            for i, point in enumerate(self.starts):
                fe = float(self.compute_free_energy(point))
                initial_energies.append((fe, i, point))
            
            initial_energies.sort(reverse=True)
            n_best = min(10, len(self.starts))
            start_points = [self.starts[idx] for _, idx, _ in initial_energies[:n_best]]
            
            print(f"\nTop {n_best} starting points:")
            for i, (fe, _, point) in enumerate(initial_energies[:n_best]):
                print(f"{i+1}. θ = [{point[0]:.4f}, {point[1]:.4f}], FE = {fe:.4f}")
        else:
            start_points = self.starts
        
        print(f"\n=== Starting Fine Optimization ===")
        for i, start_point in enumerate(start_points):
            print(f"\n{'='*50}")
            print(f"Starting optimization from point {i+1}/{len(start_points)}")
            print(f"Initial θ = [{start_point[0]:.4f}, {start_point[1]:.4f}]")
            
            theta = start_point
            m = jnp.zeros_like(theta)
            v = jnp.zeros_like(theta)
            
            fe_init = self.compute_free_energy(theta)
            grad_init = self.compute_gradient(theta)
            print(f"Initial Free Energy: {fe_init:.4f}")
            print(f"Initial Gradient: [{grad_init[0]:.4f}, {grad_init[1]:.4f}]")
            
            prev_fe = -float('inf')
            n_similar = 0
            
            for t in range(1, maxiter + 1):
                theta, m, v, gradient, grad_norm = self._adam_step(
                    theta, m, v, t, beta1, beta2, epsilon)
                curr_fe = float(self.compute_free_energy(theta))
                
                if t % log_every == 0 or t == 1:
                    sigma2, gamma = jnp.exp(theta[0]), jnp.exp(theta[1])
                    print(f"Iteration {t}:")
                    print(f"  θ = [{theta[0]:.6f}, {theta[1]:.6f}]")
                    print(f"  σ² = {sigma2:.6e}, γ = {gamma:.6e}")
                    print(f"  Free Energy = {curr_fe:.6f}")
                    print(f"  Gradient norm = {grad_norm:.6f}")

                if t > 1:
                    rel_change = abs(curr_fe - prev_fe) / (abs(prev_fe) + 1e-8)
                    if rel_change < tol/10:
                        n_similar += 1
                    else:
                        n_similar = 0
                    
                    # Stricter convergence criteria
                    if (rel_change < tol and 
                        grad_norm < 0.1 and  # Reduced from 1.0 to 0.1
                        n_similar >= 10):    # Increased from 5 to 10
                        print(f"\nConverged after {t} iterations!")
                        print(f"Final relative change: {rel_change:.2e}")
                        print(f"Final gradient norm: {grad_norm:.2e}")
                        break
                    elif t >= maxiter:
                        print("\nReached maximum iterations!")
                        print(f"Final relative change: {rel_change:.2e}")
                        print(f"Final gradient norm: {grad_norm:.2e}")
                
                prev_fe = curr_fe
            
            result = {
                'x': theta,
                'success': True,
                'nit': t,
                'free_energy': curr_fe,
                'final_gradient_norm': grad_norm
            }
            
            if curr_fe > best_fe:
                best_fe = curr_fe
                best_result = result
                print("→ New best result found!")

        print("\n=== Optimization Complete ===")
        print(f"Best result:")
        print(f"θ = [{best_result['x'][0]:.6f}, {best_result['x'][1]:.6f}]")
        print(f"Free Energy = {best_fe:.6f}")
        print(f"Final gradient norm = {best_result['final_gradient_norm']:.6f}")
        
        self.theta_opt = best_result['x']

        # Calcul du terme de complexité
        H = self.compute_hessian(self.theta_opt)
        try:
            Sigma_lambda = jnp.linalg.inv(-H)  # Inverse de la Hessienne négative
            sign, logdet = jnp.linalg.slogdet(Sigma_lambda @ self.Pi)
            complexity_term = 0.5 * logdet
            total_free_energy = best_fe + complexity_term
            
            print("\n=== Final Free Energy with Complexity Term ===")
            print(f"Initial Free Energy = {best_fe:.6f}")
            print(f"Complexity Term = {complexity_term:.6f}")
            print(f"Total Free Energy = {total_free_energy:.6f}")
        except:
            print("\nWarning: Could not compute complexity term")
            total_free_energy = best_fe

        return {
            'theta_opt': self.theta_opt,
            'free_energy': best_fe,
            'total_free_energy': total_free_energy,
            'success': best_result['success'],
            'gradient_norm': best_result['final_gradient_norm']
        }

    def solve(self, theta=None):
        if theta is None:
            if not hasattr(self, 'theta_opt'):
                raise ValueError("Il faut d'abord appeler optimizer.optimize() ou fournir theta.")
            theta = self.theta_opt

        Sigma_th = self.build_sigma_theo(theta)
        inv_Sigma_th = jnp.linalg.inv(Sigma_th)

        log_sigma2, log_gamma = theta
        gamma = jnp.exp(log_gamma)

        a_post_reduced = gamma * (self.L_reduced.T @ (inv_Sigma_th @ self.Y_reduced))

        return a_post_reduced