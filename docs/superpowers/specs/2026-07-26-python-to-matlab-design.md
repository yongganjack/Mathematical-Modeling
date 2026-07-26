# Question 1–5 Python-to-MATLAB Conversion Design

## Scope

Convert every Python module in `2025国赛/A/question1` through
`2025国赛/A/question5` into a standalone MATLAB project at
`C:/Users/Jackm/Desktop/matlab_work/2025国赛/A`.  Preserve the solver,
evaluation, plotting, JSON/CSV output, and Excel-template export behavior.
The source Python project is read-only for this work.

## Project Layout

The MATLAB project contains `question1` through `question5` directories.
Each contains a `run_questionN.m` entry point and focused function files for
data loading, model calculations, evaluation, optimization, exports, and
visualization.  Shared physical-model functions live in `question1` and are
made available to later questions by adding all question directories to the
MATLAB path from the entry points.

## Data and Compatibility

`jsondecode` reads the existing JSON configurations.  MATLAB structs replace
Python dataclasses and dictionaries; numeric arrays replace NumPy arrays.
All public functions accept and return MATLAB-native values.  Python's
zero-based entity indices are translated at data boundaries so MATLAB's
one-based indexing does not change missile/UAV semantics.

Each run creates a timestamped output directory and writes matching JSON,
CSV, figures, and, for questions 3–5, populated copies of the supplied Excel
templates.  File discovery uses paths relative to the MATLAB project and
allows an explicit configuration path.

## Solvers

Implement PSO, differential evolution, and integer PSO as local MATLAB
functions.  This avoids requiring the Global Optimization Toolbox.  Random
seeds and the runtime configuration control candidate counts and iteration
budgets.  Numerical evaluation preserves the original coverage, interval
merging, feasibility, and objective conventions.

## Error Handling and Verification

Validate required configuration fields, numerical dimensions, time bounds,
and template availability with descriptive MATLAB errors.  Verify the port by
running each `run_questionN` with the quick configuration, checking output
files, checking Excel exports for questions 3–5, and comparing deterministic
model/coverage values against the Python implementation within floating-point
tolerance.  Optimizer end points are assessed for feasibility and objective
quality rather than exact equality because stochastic trajectories differ by
runtime.

## Deliberate Non-Goals

The port does not modify Python source, migrate Python tests verbatim, require
third-party MATLAB toolboxes, or reproduce Python CLI flags exactly.  MATLAB
entry-point name-value arguments cover the required runtime settings.
