# POTline

This Python framework is designed to run a pipeline to train Machine Learning Interatomic Potentials (MLIAP) and benchmark their inference time and accuracy on calculating mechanical properties in LAMMPS.

## Features

- Train MLIAP models
- Benchmark inference time
- Evaluate accuracy on mechanical properties in LAMMPS (using [Potential_benchmark_iron](https://github.com/leiapple/Potential_benchmark_iron))

## Installation

To install the framework and its dependencies, follow these steps:

1. Create and activate a new Conda environment with the requirements of your choosen model. Currently supported models are:
    - [PACE](https://github.com/ICAMS/python-ace?tab=readme-ov-file) --> environment should be named `pl`
    - [MACE](https://github.com/ACEsuit/mace) (no properties simulation) --> environment should be named `mace`
    - [grACE](https://github.com/ICAMS/grace-tensorpotential) --> environment should be named `grace`
In the same environment install [simple_slurm](https://github.com/amq92/simple_slurm) and [XPOT](https://github.com/dft-dutoit/XPOT) from GIT. Do not use the PyPi versions.
2. Clone the repository and open it:
    ```bash
    git clone https://github.com/TheNewPing/POTline.git
    cd POTline
    ```
3. Install LAMMPS accordingly to the model's documentation. Ensure to enable GPU support.
You can also use the Slurm scripts `install_lammps_modelname.sh` to start the installation after having completed the LAMMPS setup. Note that they are tailored for [Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius), adjust them to your needs.
    - [PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/quickstart/#lammps)
    - [MACE](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html)
    - [grACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/#lammps-with-grace)

## Usage
CURRENTLY ONLY USAGE VIA SLURM IS BEIGN TESTED.

To use the POTline framework, you can run the `run.py` script with various command line arguments to control its behavior. Below are the available options:

```bash
python src/run.py --config <path_to_config> [options]
```

### Arguments

- `--config`: Path to the config file (example: `src/data/pace.hjson`)

### Options

- `--nohyper`: Disable potential fitting (requires `--fitted` to work)
- `--nodeep`: Disable fitting on best models from hyperparameter optimization
- `--noconversion`: Disable LAMMPS potential conversion
- `--noinference`: Disable inference benchmark
- `--noproperties`: Disable properties simulation

### Configuration File Syntax

The configuration file for POTline is written in HJSON format, which is a user-friendly extension of JSON. Below is a description of the main sections and their respective parameters:

#### General
- `lammps_bin_path`: Path to the LAMMPS binary.
- `model_name`: Name of the model (currently supports `pacemaker, mace, gracemaker`).
- `best_n_models`: Number of best models to use in inference and simulation step.
- `hpc`: HPC mode, keep always True.
- `cluster`: Cluster configuration to use (currently supports only `snellius`).
- `sweep_path`: Output path for the experiments

### Deep training
- `max_epochs`: Max number of epochs for deeper training on best models.

#### Inference
- `prerun_steps`: Number of pre-run steps.
- `max_steps`: Maximum number of steps.

#### Data Analysis
- `lammps_inps_path`: Path to LAMMPS input files (from Potential_benchmark_iron).
- `pps_python_path`: Path to Python scripts for post-processing (from Potential_benchmark_iron).
- `ref_data_path`: Path to reference data (from Potential_benchmark_iron).
- `email`: Address used to send simulation results.

#### Hyperparamerter optimization
- `max_iter`: Number of iterations of ask-tell for the baesyan optimizer.
- `n_initial_points`: Consult `skopt.Optimizer`.
- `n_points`: Number of parameters sets asked at each iteration to the optimizer.
- `strategy`: Strategy for the optimizer, consult `skopt.Optimizer`.
- `energy_weight`: Loss weight of the energy component (0.0 - 1.0).
- `optimizer_params`: Model-specific configuration:
    - [PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/inputfile/)
    - [MACE](https://mace-docs.readthedocs.io/en/latest/guide/training.html)
    - [grACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/inputfile/)

## TODO

- ~~Implement basic pipeline with XPOT, Pacemaker, and BCC iron integration~~
- ~~Enable parallel hyperparameter optimizaion~~
- ~~Add another training phase on the best models from hyperparameter optimization~~
- ~~Add MACE support~~
- ~~Add GRACE support~~
- Add MACE properties simulation
- Refactor (again) codebase
- Add Slurm fault tolerance
- Add optimizer checkpoints
- Add more properties simulations
- Add summary data presentation
- Test more model variants
- Test finetuning where available
- Test new architectures
- Reintroduce Slurm-less pipeline (SoonTM)
