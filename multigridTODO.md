# 📊 Multigrid Efficiency Analysis – Pure Poisson (MMS)

This project evaluates the performance of multigrid methods (V-cycle, W-cycle, FMG) for solving the 2D Poisson equation using a Method of Manufactured Solutions (MMS) setup. The goal is to identify the optimal combination of smoother, smoothing schedule, and multigrid cycle type based on both **accuracy** and **computational efficiency** (time and FLOPs).

---

## 📁 Project Structure

```
configs/                     # YAML config files for each test
src/                         # Solver implementation
results/                     # Output data: timing, residuals, errors
scripts/                     # Automation and plotting
README.md                    # This file
```

---

## 🔧 Setup

### 📦 Requirements

- Python 3.9+
- NumPy
- Matplotlib
- PyYAML
- [`pypapi`](https://pypi.org/project/pypapi/) – for FLOP measurements

Install locally:
```bash
pip install numpy matplotlib pyyaml pypapi
```

> On DTU HPC, load PAPI:
```bash
module load papi
```

---

## 🧪 Problem Setup – MMS Poisson

We solve:

\[
-\Delta u = f \quad \text{in } [0,1]^2, \quad u(x,y) = \sin(\pi x)\sin(\pi y)
\]

The manufactured source term is:

\[
f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)
\]

Use 2nd-order central differences on a uniform grid.

---

## 🔷 PART 1 – V-CYCLE ONLY

### 🔁 What You Vary

- **Smoothers**: Jacobi, Gauss–Seidel, Red–Black Gauss–Seidel
- **Smoothing steps**: \( (\nu_1, \nu_2) \in \{(1,1), (2,1), (1,2), (2,2)\} \)
- **Grid size**: Start with \(64 \times 64\), confirm on larger grids

### 📈 What You Measure

- Residual norm vs iteration
- L2 error vs analytical solution
- **Wall time**
- **FLOPs** using `pypapi`
- Derived **Work Units**:
  - Naive WU: #Unknowns × (ν₁ + ν₂) × cycles
  - FLOP-based WU: from `pypapi` count

### ✅ What You Conclude

- Best (smoother, ν₁, ν₂) combination for minimal **time-to-error**
- Whether faster convergence (fewer cycles) offsets more expensive smoothers
- Whether FLOP-based WU aligns with wall time

---

## 🔷 PART 2 – CYCLE TYPE COMPARISON (V vs W vs FMG)

### 🧪 What You Fix

- Use best smoother + ν₁/ν₂ from Part 1

### 🔁 What You Vary

- **Cycle type**: V-cycle, W-cycle, FMG
- **Grid sizes**: \(32^2, 64^2, 128^2, 256^2\)

### 📈 What You Measure

- L2 error after convergence
- Time to reach truncation-level error
- Total FLOPs via `pypapi`
- Residual norm (for V/W)
- Number of multigrid cycles (V/W only)

### ✅ What You Conclude

- Which multigrid cycle type gives lowest **time-to-accuracy**
- FMG efficiency: does it reach truncation error in one cycle?
- Whether W-cycle is worth extra cost vs V-cycle

---

## 📊 Suggested Plots

- Residual norm vs iteration (log scale)
- L2 error vs wall time
- L2 error vs FLOPs
- Work units vs L2 error (naive + FLOP-based)
- Time/FLOPs breakdown per method

---

## 🧠 Final Deliverables

- Optimal smoother and smoothing schedule (from Part 1)
- Optimal multigrid cycle (from Part 2)
- CSV logs of all runs
- Clean reproducible YAML configs
- Matplotlib or LaTeX-ready plots for report/poster
