# POTline

This Python framework is designed to run a pipeline to train Machine Learning Interatomic Potentials (MLIAP) and benchmark their inference time and accuracy on calculating mechanical properties in LAMMPS.

## Features

- Train MLIAP models
- Benchmark inference time
- Evaluate accuracy on mechanical properties in LAMMPS (using [Potential_benchmark_iron](https://github.com/leiapple/Potential_benchmark_iron))

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

## Usage
CURRENTLY ONLY USAGE VIA SLURM IS SUPPORTED.

To use the POTline framework, you can run the `main.py` script with various command line arguments to control its behavior. Below are the available options:

```bash
python main.py --config <path_to_config> --iterations <num_iterations> [options]
```

### Arguments

- `--config`: Path to the config file (default: `src/data/config.hjson`)
- `--iterations`: Number of optimizer iterations to run (default: `1`)

### Options

- `--nohyper`: Disable potential fitting (requires `--fitted` to work)
- `--nodeep`: Disable fitting on best models from hyperparameter optimization
- `--noconversion`: Disable LAMMPS potential conversion
- `--noinference`: Disable inference benchmark
- `--noproperties`: Disable properties simulation
- `--nohpc`: Disable HPC mode
- `--fitted`: Path to the fitted potential (do not use without `--nohyper`)

### Configuration File Syntax

The configuration file for POTline is written in HJSON format, which is a user-friendly extension of JSON. Below is a description of the main sections and their respective parameters:

#### General
- `lammps_bin_path`: Path to the LAMMPS binary.
- `model_name`: Name of the model (currently supports only `pacemaker`).
- `best_n_models`: Number of best models to use in inference and simulation step.

### Deep training
- `max_epochs`: Max number of epochs for deeper training on best models.

#### Inference
- `prerun_steps`: Number of pre-run steps.
- `max_steps`: Maximum number of steps.
- `n_cpu`: Number of CPUs to use.

#### Data Analysis
- `lammps_inps_path`: Path to LAMMPS input files (from Potential_benchmark_iron).
- `pps_python_path`: Path to Python scripts for post-processing (from Potential_benchmark_iron).
- `ref_data_path`: Path to reference data (from Potential_benchmark_iron).

#### Hyperparamerter optimization
- `max_iter`: Number of iterations of ask-tell for the baesyan optimizer.
- `n_initial_points`: Consult `skopt.Optimizer`.
- `n_points`: Number of parameters sets asked at each iteration to the optimizer.
- `xpot`: Consult [XPOT](https://github.com/dft-dutoit/XPOT)
- Model-specific configuration:
    - [pacemaker](https://pacemaker.readthedocs.io/en/latest/pacemaker/inputfile/)

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
- ~~Enable parallel hyperparameter optimizaion~~
- ~~Add another training phase on the best models from hyperparameter optimization~~
- Add MACE support
- Add GRACE support
- Reintroduce Slurm-less pipeline
