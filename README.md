# Discrete-Time LQG Regulator

This repository contains a Python implementation of a discrete-time Linear-Quadratic-Gaussian (LQG) regulator for an infinite-horizon linear system.

## Mathematical Formulation

The discrete-time linear system is defined as:
$$x_{k+1} = A x_k + B u_k + w_k$$
$$y_k = C x_k + v_k$$

where $w_k$ and $v_k$ represent independent discrete-time Gaussian white noise processes with covariance matrices $W$ and $V$.

### Controller & Estimator
The optimal control law is given by:
$$u_k = -F \hat{x}_k$$

The feedback gain $F$ and the Kalman gain $L$ are computed by solving their respective discrete-time algebraic Riccati equations (DARE):
$$F = (B^T S B + R)^{-1} B^T S A$$
$$L = P C^T (C P C^T + V)^{-1}$$

### Expected Steady-State Cost
The script calculates both the empirical instantaneous stage cost and the theoretical expected limit. By defining an augmented closed-loop state vector $Z_k = [x_k^T, \hat{x}_{k|k-1}^T]^T$, the steady-state covariance $Z_{cov}$ is found via the discrete Lyapunov equation:
$$Z_{cov} = A_{cl} Z_{cov} A_{cl}^T + B_{cl} \text{diag}(W, V) B_{cl}^T$$

The theoretical expected stage cost is then extracted from $Z_{cov}$:
$$\mathbb{E}[L_k] = \text{Tr}(Q \Sigma_x) + \text{Tr}(R \Sigma_u)$$

## Usage
Run the script to compute the optimal gains, simulate the system, and visualize the true vs. estimated states alongside the stage cost evaluated against the theoretical steady-state limit.

```bash
python LQG.py
