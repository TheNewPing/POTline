# POTline

This Python framework is designed to run a pipeline to train Machine Learning Interatomic Potentials (MLIAP) and benchmark their inference time and accuracy on calculating mechanical properties in LAMMPS.

## Features

- Train MLIAP models
- Benchmark inference time
- Evaluate accuracy on mechanical properties in LAMMPS

## Installation

To install the framework and its dependencies, follow these steps:

1. `Optional` Create and activate a new Conda environment:
    ```bash
    conda create --name potline python=3.10
    conda activate potline
    ```
2. Clone the repository and open it:
    ```bash
    git clone https://github.com/TheNewPing/POTline.git
    cd POTline
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Install [XPOT](https://github.com/dft-dutoit/XPOT)
5. Install [Potential_benchmark_iron](https://github.com/leiapple/Potential_benchmark_iron)

## Usage
To use the POTline framework, you can run the `main.py` script with various command line arguments to control its behavior. Below are the available options:

```bash
python main.py --config <path_to_config> --iterations <num_iterations> [options]
```

### Arguments

- `--config`: Path to the config file (default: `src/data/config.hjson`)
- `--iterations`: Number of optimizer iterations to run (default: `1`)

### Options

- `--nofitting`: Disable potential fitting (requires `--fitted` to work)
- `--noconversion`: Disable LAMMPS potential conversion
- `--noinference`: Disable inference benchmark
- `--noproperties`: Disable properties simulation
- `--nohpc`: Disable HPC mode
- `--fitted`: Path to the fitted potential

### Configuration File Syntax

The configuration file for POTline is written in HJSON format, which is a user-friendly extension of JSON. Below is a description of the main sections and their respective parameters:

#### General
- `lammps_bin_path`: Path to the LAMMPS binary.
- `out_yace_path`: Directory with LAMMPS potentials (used only with `--noconversion`).
- `model_name`: Name of the model (currently supports only `pacemaker`).
- `best_n_models`: Number of best models to use in inference and simulation step.

#### Inference
- `prerun_steps`: Number of pre-run steps.
- `max_steps`: Maximum number of steps.
- `n_cpu`: Number of CPUs to use.

#### Data Analysis
- `lammps_inps_path`: Path to LAMMPS input files (from Potential_benchmark_iron).
- `pps_python_path`: Path to Python scripts for post-processing (from Potential_benchmark_iron).
- `ref_data_path`: Path to reference data (from Potential_benchmark_iron).

#### Optimizer
- `xpot`: Configuration for XPOT optimizer.
    - `project_name`: Name of the project.
    - `sweep_name`: Name of the sweep.
    - `error_method`: Method to calculate error (e.g., RMSE).
    - `alpha`: Alpha parameter for the optimizer.
- `cutoff`: Cutoff value.
- `seed`: Random seed.
- `metadata`: Metadata for the optimization.
    - `purpose`: Purpose of the optimization.
- `data`: Data configuration.
    - `filename`: Path to the dataset.
    - `test_size`: Proportion of data to use for testing.
- `potential`: Potential configuration.
    - `deltaSplineBins`: Delta spline bins value.
    - `elements`: List of elements.
    - `embeddings`: Embedding configurations.
    - `rankmax`: Maximum rank.
    - `bonds`: Bond configurations.
    - `functions`: Function configurations.
- `fit`: Fit configuration.
    - `loss`: Loss function parameters.
    - `optimizer`: Optimizer to use.
    - `maxiter`: Maximum number of iterations.
    - `repulsion`: Repulsion parameter.
    - `trainable_parameters`: Parameters that are trainable.
- `backend`: Backend configuration.
    - `evaluator`: Evaluator to use.
    - `batch_size`: Batch size.
    - `batch_size_reduction`: Whether to reduce batch size.
    - `batch_size_reduction_factor`: Factor by which to reduce batch size.
    - `display_step`: Step interval for displaying progress.
    - `gpu_config`: GPU configuration.

This configuration file allows you to customize various aspects of the POTline framework to suit your specific needs.

### Example Usage

```bash
python main.py --config src/data/config.hjson --iterations 10 --nofitting --fitted path/to/fitted/potential
```

### Slurm support

To run the POTline framework using Slurm, you can use the provided Slurm script located at `src/hpc/potline_hpc.txt`. The current configuration is tailored for the Snellius cluster. You may need to adapt the script to match the specifications and configurations of the cluster you are using.

```bash
sbatch src/hpc/potline_hpc.txt
```

## TODO

- ~~Implement basic pipeline with XPOT, Pacemaker, and BCC iron integration~~
- Add MACE support
- Add GRACE support
