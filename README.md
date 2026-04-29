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

## Usage

Run the script to compute the optimal gains, simulate the closed-loop system, and visualize the true versus estimated states:

```bash
python LQG.py
