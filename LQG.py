import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def solve_dlqg(A, B, C, Q, R, W, V):
    """Solves the infinite-horizon discrete-time LQG problem."""
    S = la.solve_discrete_are(A, B, Q, R)
    F = la.inv(B.T @ S @ B + R) @ (B.T @ S @ A)
    
    P = la.solve_discrete_are(A.T, C.T, W, V)
    L = P @ C.T @ la.inv(C @ P @ C.T + V)
    
    return F, L

def compute_theoretical_cost(A, B, C, Q, R, W, V, F, L):
    """Computes the expected steady-state stage cost via the discrete Lyapunov equation."""
    n = A.shape[0]
    
    # Formulate augmented closed-loop matrices
    A_cl = np.block([
        [A - B @ F @ L @ C, -B @ F @ (np.eye(n) - L @ C)],
        [(A - B @ F) @ L @ C, (A - B @ F) @ (np.eye(n) - L @ C)]
    ])
    
    B_cl = np.block([
        [np.eye(n), -B @ F @ L],
        [np.zeros((n, n)), (A - B @ F) @ L]
    ])
    
    W_aug = la.block_diag(W, V)
    
    # Solve discrete Lyapunov equation
    Z_cov = la.solve_discrete_lyapunov(A_cl, B_cl @ W_aug @ B_cl.T)
    
    # Extract sub-covariances and compute trace matrices
    Z11 = Z_cov[:n, :n]
    
    K_x = -F @ L @ C
    K_xhat = -F @ (np.eye(n) - L @ C)
    K_v = -F @ L
    K_aug = np.block([K_x, K_xhat])
    
    E_x_cost = np.trace(Q @ Z11)
    E_u_cost = np.trace(R @ (K_aug @ Z_cov @ K_aug.T + K_v @ V @ K_v.T))
    
    return E_x_cost + E_u_cost

def main():
    # 1. 3x3 System Definition
    A = np.array([[1.05, 0.1, 0.0],
                  [0.0,  0.95, 0.1],
                  [0.0,  0.0,  0.90]])
    B = np.array([[0.0], 
                  [0.0], 
                  [1.0]])
    C = np.array([[1.0, 0.0, 0.0]])
    
    # 2. Aggressive Weight Tuning
    Q = np.diag([100.0, 50.0, 10.0]) 
    R = np.array([[0.01]])           
    W = np.diag([0.1, 0.1, 0.1])
    V = np.array([[0.05]])
    
    # 3. Compute Gains & Theoretical Limit
    F, L = solve_dlqg(A, B, C, Q, R, W, V)
    expected_cost = compute_theoretical_cost(A, B, C, Q, R, W, V, F, L)
    print(f"Theoretical Expected Steady-State Cost: {expected_cost:.4f}")

    # 4. Simulation Setup (Extended horizon to show steady-state)
    steps = 150
    x_true = np.zeros((3, steps))
    x_est = np.zeros((3, steps))
    
    x_true[:, 0] = [1.0, -0.5, 0.0]
    x_est[:, 0]  = [0.0, 0.0, 0.0]
    stage_costs = np.zeros(steps)
    
    # 5. Simulation Loop
    for k in range(steps - 1):
        w_k = np.random.multivariate_normal(np.zeros(3), W)
        v_k = np.random.multivariate_normal(np.zeros(1), V)
        
        y_k = C @ x_true[:, k] + v_k
        
        # Filter and Control
        x_est[:, k] = x_est[:, k] + L.flatten() * (y_k - C @ x_est[:, k])
        u_k = -F @ x_est[:, k]
        
        # Stage Cost Evaluation
        cost_k = x_true[:, k].T @ Q @ x_true[:, k] + u_k.T @ R @ u_k
        stage_costs[k] = cost_k.item() if hasattr(cost_k, 'item') else cost_k
        
        # Propagation
        x_true[:, k+1] = A @ x_true[:, k] + B @ u_k + w_k
        x_est[:, k+1]  = A @ x_est[:, k] + B @ u_k

    # Final step cost
    final_cost = x_true[:, -1].T @ Q @ x_true[:, -1]
    stage_costs[-1] = final_cost.item() if hasattr(final_cost, 'item') else final_cost

    # 6. Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: State Regulation
    ax1.plot(x_true[0, :], label='True State $x_1$')
    ax1.plot(x_est[0, :], '--', label='Estimated State $\hat{x}_1$')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax1.set_title('Discrete-Time LQG Regulation')
    ax1.set_ylabel('State Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Instantaneous Objective Function
    ax2.plot(stage_costs, 'g-', label='Empirical Stage Cost $L_k$', alpha=0.7)
    ax2.axhline(expected_cost, color='red', linestyle='--', linewidth=2, 
                label=f'Theoretical Limit ($\mathbb{{E}}[L_k]$ = {expected_cost:.1f})')
    ax2.set_title('Instantaneous Stage Cost vs. Theoretical Limit')
    ax2.set_xlabel('Time Step $k$')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()