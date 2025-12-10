# Project File Summary

This document provides a detailed description of each significant file within the Vesta project.

## Root Directory Files

### `.gitignore`

This file specifies intentionally untracked files and directories that Git should ignore. It includes common build artifacts (`*.lqxo`, `*.out`, `*.xml`), log files (`*.log`, `/logs`), virtual environment directories (`/venv`), IDE-specific configuration (`/.idea`), temporary Python notebooks (`/tmp.ipynb`), and intermediate data files (`/bench_smt_train.mat`, `/bench_smt_test.mat`). Its purpose is to keep the repository clean and focused on source code.

### `GEMINI.md`

This Markdown file provides an overview of the Vesta project specifically for the Gemini AI. It summarizes the project's purpose, main technologies, and architecture, and provides instructions on how to build, run benchmarks, and execute tests. It also outlines development conventions, including code structure and how optimization problems are defined, serving as a comprehensive initial context for AI interactions.

### `README.md`

This is the primary documentation file for the project, offering a high-level introduction. It identifies the project as "VESTA: Versatile and Efficient Service Topology Allocation," briefly describing its core purpose and providing an entry point for understanding the project's goals.

### `requirements.txt`

This file lists all Python package dependencies required to run the Vesta project. It includes specific version pins for some libraries (e.g., `line-solver==2.0.31.10`) and allows for broader compatibility with others. Key dependencies cover numerical computing (`numpy`, `scipy`, `pandas`), plotting (`matplotlib`), graph theory (`networkx`), and various optimization/Hyperparameter Optimization (HPO) frameworks (`openbox`, `nevergrad`, `opytimizer`, `scikit-opt`, `smt`, `pyDOE3`).

## `bench/` Directory Files

### `bench/bench_diary.txt`

This plain text file functions as a development log or diary, capturing observations and performance notes about different optimization algorithms as they are benchmarked against Vesta's problems. It offers qualitative insights into the behavior, relative strengths, and limitations of algorithms such as `differential_evolution`, `SMT` EGO, `COBYLA`, `SLSQP`, `OpenBox`, `Nevergrad`, and `Opytimizer`.

### `bench/bench_ego.py`

This benchmark script utilizes the EGO (Efficient Global Optimization) algorithm from the `SMT` (Surrogate Modeling Toolbox) library to optimize the `opt_lqn_2` problem defined in `vesta/problems.py`. It configures a 2-variable continuous design space, performs initial data sampling, and then executes the EGO optimizer. The objective function `fun` wraps `opt_lqn_2` and incorporates a penalty term to guide the optimization process.

### `bench/bench_ego20.py`

Similar to `bench_ego.py`, this script benchmarks the `SMT` EGO algorithm, but targets the `opt_lqn_20` problem, a 20-variable mixed-integer optimization challenge. It establishes a complex design space comprising both float and integer variables, conducts initial data sampling, and then runs the EGO optimizer. The objective function `fun` integrates `opt_lqn_20` with a penalty.

### `bench/bench_ego4.py`

This script benchmarks the `SMT` EGO algorithm against the `opt_lqn_4` problem, a 4-variable mixed-integer optimization task. It defines a design space incorporating both float and integer variables, performs initial sampling, and subsequently executes the EGO optimizer.

### `bench/bench_ego8.py`

This script benchmarks the `SMT` EGO algorithm for the `opt_lqn_8` problem, an 8-variable mixed-integer optimization scenario. It sets up a design space with both continuous and integer variables, carries out data sampling, and then runs the EGO optimizer.

### `bench/bench_hybrid_baseline.py`

This script serves as a baseline benchmark, employing `scipy.optimize.differential_evolution` to directly solve a 4-variable mixed-integer optimization problem (`opt_lqn_4`). It specifies bounds and integrality constraints for the variables and measures the execution time, providing a reference point for more complex hybrid optimization strategies.

### `bench/bench_hybrid.py`

This script showcases a hybrid optimization strategy by nesting `scipy.optimize` methods. It features an `inner_opt` function that uses `minimize` (e.g., SLSQP or COBYLA) for continuous variables and an `outer_opt` function that employs `differential_evolution` for integer variables. This combination facilitates both local and global search, and it's applied to the `opt_lqn_4` problem.

### `bench/bench_nevergrad.py`

This script benchmarks an optimization problem using the `nevergrad` library. It attempts to optimize the `opt_lqn_2` problem, defining constraints and utilizing `nevergrad`'s `NGOpt` optimizer to identify an optimal solution. The script's comments indicate a potential issue related to `topology.solve()`.

### `bench/bench_openbox.py`

This script benchmarks an optimization problem using the `OpenBox` library. It defines the search space for the `opt_lqn_2` problem using `OpenBox`'s `sp.Space` and `sp.Real` objects. It then initializes and executes the `OpenBox.Optimizer`, specifying the objective function (`fun`), the number of constraints, and the maximum number of runs.

### `bench/bench_opytimizer.py`

This script benchmarks an optimization problem utilizing the `Opytimizer` framework, specifically implementing the Differential Evolution (DE) algorithm. It defines an objective function `fun` that wraps `opt_lqn_2` with an added penalty term, establishes the search space with appropriate bounds, and then runs the `Opytimizer` with the `DE` optimizer.

### `bench/bench_scipy_cobyla.py`

This script benchmarks the `COBYLA` (Constrained Optimization BY Linear Approximation) algorithm from `scipy.optimize` to solve a 2-variable continuous optimization problem (`opt_lqn_2`). It defines the objective function `fun`, variable bounds, and inequality constraints, subsequently invoking `scipy.optimize.minimize` with the `COBYLA` method.

### `bench/bench_scipy_de.py`

This script benchmarks the `differential_evolution` algorithm from `scipy.optimize` for a 2-variable continuous optimization problem (`opt_lqn_2`). It specifies the objective function `fun`, variable bounds, and linear constraints before executing the `differential_evolution` solver.

### `bench/bench_scipy_de8.py`

This script benchmarks the `differential_evolution` algorithm from `scipy.optimize` for an 8-variable mixed-integer optimization problem (`opt_lqn_8`). It defines the objective function `fun`, variable bounds, and linear constraints, including explicit integrality specifications for certain variables, and then runs `differential_evolution`.

### `bench/bench_scipy_slsqp.py`

This script benchmarks the `SLSQP` (Sequential Least Squares Programming) algorithm from `scipy.optimize` for a 2-variable continuous optimization problem (`opt_lqn_2`). It defines the objective function `fun`, variable bounds, and inequality constraints, and then calls `scipy.optimize.minimize` with the `SLSQP` method.

### `bench/bench_sko_de.py`

This script benchmarks the Differential Evolution (DE) algorithm provided by the `scikit-opt` (`sko`) library. It optimizes the `opt_lqn_2` problem, defining the objective function, problem dimensionality, population size, maximum iterations, lower and upper bounds, and equality constraints before executing the `sko.DE` optimizer.

### `bench/bench_smt.ipynb`

This Jupyter Notebook is dedicated to benchmarking various surrogate models from the `SMT` (Surrogate Modeling Toolbox) library for approximating the `opt_lqn_2` problem. It outlines the process of defining the objective function, generating training and test datasets using LHS (Latin Hypercube Sampling), and evaluating different surrogate models (e.g., RBF, KRG, KPLS) based on their training efficiency and prediction accuracy (MAPE, MaxAPE). The notebook also includes a section for using the best-performing surrogate model within a `scipy.optimize.minimize` routine for optimization.

### `bench/bench_smt20.ipynb`

This Jupyter Notebook extends the `SMT` surrogate model benchmarking to the `opt_lqn_20` problem, a more complex 20-variable optimization task that includes both float and integer types. It details the setup of the mixed-integer design space, the generation of training data, and attempts to train and evaluate various `SMT` surrogate models. The notebook's output often indicates a `ValueError` during surrogate model training, suggesting challenges with data dimensionality or model configuration for this specific problem. It also includes an optimization step utilizing `differential_evolution` with a surrogate function.

### `bench/bench_smt8.ipynb`

This Jupyter Notebook is analogous to `bench_smt.ipynb` and `bench_smt20.ipynb`, but specifically focuses on the `opt_lqn_8` problem. It defines the 8-dimensional mixed-integer design space, generates training and test datasets, and evaluates the performance of `SMT` surrogate models. The notebook attempts to perform optimization using `differential_evolution` with a surrogate model, with output indicating potential issues during the `differential_evolution` process, possibly related to `inf` values for `f(x)` in early iterations.

## `tests/` Directory Files

### `tests/dag.ipynb`

This Jupyter Notebook (`.ipynb`) file serves as an exploratory or development environment for creating and manipulating Directed Acyclic Graphs (DAGs) using the `networkx` library, similar to its Python script counterpart. It contains executable code cells for importing necessary libraries, defining a `create_random_dag` function, and adding attributes to graph nodes and edges. The notebook's output includes debugging information and may show `ValueError` messages related to graph layout or visualization, suggesting it's a work-in-progress or example notebook.

### `tests/dag.py`

This script provides utility functions for creating and manipulating random Directed Acyclic Graphs (DAGs) using the `networkx` library. It includes `create_random_dag` for generating graphs with a specified number of nodes and edge probability, `add_edge_attributes` for assigning properties like selection probabilities to edges, and `add_node_attributes` for adding numerical attributes to nodes. The script also contains code for visualizing the generated DAGs using `matplotlib.pyplot`. This file likely serves as a helper or an example for managing graph structures pertinent to service topologies within the Vesta framework.

## `vesta/` Directory Files

### `vesta/__init__.py`

This is the initialization file for the `vesta` Python package, making its contents directly available when the package is imported. It imports core components from `line_solver` and `networkx`. The file defines several key classes essential for constructing service topology models:
-   `ServiceTopology`: Extends `line_solver.LayeredNetwork`, providing a representation of a system's topology and a `solve` method to compute performance metrics via `SolverLQNS`.
-   `User`, `Host`, `Container`: Classes that model different entities within the service topology, inheriting from `line_solver.Processor` or `line_solver.Task`.
-   `Service`, `ServiceStep`: Classes for defining services and their constituent processing steps within containers.
-   `SynchCaller`: Represents a synchronous user or caller of services, enabling the initialization and configuration of a Directed Acyclic Graph (DAG) of service calls using `networkx`. It also includes methods to retrieve performance metrics such as throughput, response time, and utilization.
Additionally, it imports all functions from `vesta.problems`, making these optimization problem definitions directly accessible upon importing `vesta`.

### `vesta/problems.py`

This file is central to the Vesta framework, defining various optimization problems that are used for benchmarking and development. Each problem is typically implemented as a Python function (e.g., `opt_lqn_2`, `opt_lqn_4`, `opt_lqn_8`, `opt_lqn_20`, `stoch_opt_lqn_20`) that accepts an array of parameters (`x`). Within these functions, a `ServiceTopology` is constructed using `vesta` components like `Host`, `Container`, `Service` (with exponential `Exp` or Erlang `Erlang` distributions), and `SynchCaller`. After the topology is built, it is solved, and a performance metric (such as throughput) is returned. These problems represent different levels of complexity in terms of the number and types of variables involved (continuous, integer, mixed-integer).