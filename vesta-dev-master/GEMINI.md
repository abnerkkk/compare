# Project: Vesta - Optimization Framework

## Project Overview

Vesta is a Python-based framework designed for implementing and benchmarking various optimization algorithms, particularly in the context of Hyperparameter Optimization (HPO). Its core functionality revolves around defining and solving service topologies to maximize throughput or other performance metrics, often involving mixed-integer variables. The framework provides components to model hosts, containers, services, and user interactions within a service topology.

**Key Features:**
-   Definition of complex optimization problems involving service topologies.
-   Benchmarking of different HPO algorithms (e.g., OpenBox, Nevergrad, Opytimizer, SciPy, Scikit-Optimize).
-   Utilizes a custom `vesta` module for modeling system components and their interactions.

**Main Technologies:**
-   Python
-   `line-solver`: For solving system models.
-   `numpy`: Numerical operations.
-   `scipy`: Scientific computing, including optimization routines.
-   `networkx`: Graph manipulation (potentially for service dependency graphs).
-   `openbox`, `nevergrad`, `opytimizer`, `scikit-opt`, `smt`: Various optimization and HPO libraries.

## Building and Running

### Dependencies

All project dependencies are listed in `requirements.txt`. To install them, use `pip`:

```bash
pip install -r requirements.txt
```

### Running Benchmarks

The `bench/` directory contains scripts for benchmarking various optimization algorithms against defined problems. These scripts can be executed directly as Python programs.

Example:
To run the `bench_hybrid.py` benchmark:

```bash
python bench/bench_hybrid.py
```

### Running Tests

The project uses `pytest` for running tests. The test files are located in the `tests/` directory.

To execute all tests:

```bash
pytest
```

## Development Conventions

### Code Structure

-   **`vesta/`**: Contains the core framework for defining service topologies, hosts, containers, services, and callers. `problems.py` defines specific optimization problems using these core components.
-   **`bench/`**: Houses scripts for benchmarking different optimization algorithms. Each script typically focuses on a specific algorithm or problem variation.
-   **`tests/`**: Contains unit and integration tests for the project.

### Optimization Problem Definition

Optimization problems are typically defined as Python functions within `vesta/problems.py`. These functions accept an array of variables (`x`) as input, construct a `ServiceTopology` using `vesta` components based on these variables, solve the topology, and return a scalar objective value (e.g., throughput) to be optimized. Some problems involve mixed-integer variables.

### Testing

Tests are written using `pytest`. The `tests/dag.py` example suggests the use of `networkx` for graph-related structures, which might be integral to how service dependencies are managed within the framework.
