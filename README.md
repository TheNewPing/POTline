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
4. Install [XPOT](https://github.com/dft-dutoit/XPOT):
5. Install [Potential_benchmark_iron](https://github.com/leiapple/Potential_benchmark_iron):

## Usage
To use the POTline framework, you can run the `main.py` script with various command line arguments to control its behavior. Below are the available options:

```bash
python main.py --config <path_to_config> --iterations <num_iterations> [options]
```

### Arguments

- `--config`: Path to the config file (default: `src/data/config.hjson`)
- `--iterations`: Number of iterations to run (default: `1`)

### Options

- `--nofitting`: Disable potential fitting (requires `--fitted` to work)
- `--noconversion`: Disable LAMMPS potential conversion
- `--noinference`: Disable inference benchmark
- `--noproperties`: Disable properties simulation
- `--nohpc`: Disable HPC mode
- `--fitted`: Path to the fitted potential

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
