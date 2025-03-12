# Naviflow

Code for bachelors project 'Finite Volume based CFD for Lid Driven Cavity Flow'

Naviflow is a Python package for solving the lid-driven cavity problem using the finite volume method.

Functions are in the naviflow folder.

## Object-Oriented Architecture (naviflow_oo)

The object-oriented version of Naviflow is organized into three main components:

### 1. Preprocessing
- **Mesh Generation**: Create and manage computational grids
- **Field Initialization**: Set up initial conditions for all fields
- **Boundary Conditions**: Define and apply boundary conditions

### 2. Solver
- **Solvers**: Implement numerical methods for different equation types
- **Algorithms**: Coordinate the solution process (SIMPLE, etc.)

### 3. Postprocessing
- **Visualization**: Generate plots and visualizations of results
- **Validation**: Verify solution accuracy and conservation properties

### 4. Case Management
- Orchestrates the entire simulation process
- Connects preprocessing, solver, and postprocessing components

## Main scripts 

Main scripts are in the main folder.

## Installation

To install Naviflow, run the following command in root directory:

```bash
pip install -e .
```

