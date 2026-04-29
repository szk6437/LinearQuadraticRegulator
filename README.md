# Discrete-Time LQG Regulator

This repository contains a Python implementation of a discrete-time Linear-Quadratic-Gaussian (LQG) regulator for an infinite-horizon linear system.

## System Summary

This project simulates a linear system subject to independent Gaussian process and measurement noise. It implements the optimal LQG control architecture relying on the separation principle:

* **State Estimation:** A discrete-time Kalman filter computes optimal estimates of the internal system states based on noisy sensor measurements.
* **Optimal Control:** A Linear-Quadratic Regulator (LQR) calculates the control input by applying a feedback gain directly to the Kalman filter's state estimates.

Both the estimator and controller are designed for steady-state, infinite-horizon operation. The optimal Kalman gain and LQR feedback gain are computed dynamically by solving their associated discrete-time Algebraic Riccati Equations (DARE).

## Usage

Run the script to compute the optimal gains, simulate the closed-loop system, and visualize the true versus estimated states:

```bash
python LQG.py
