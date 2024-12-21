# POTline

This Python framework is designed to run a pipeline to train Machine Learning Interatomic Potentials (MLIAP) and benchmark their inference time and accuracy on calculating mechanical properties in LAMMPS (using [Potential_benchmark_iron](https://github.com/leiapple/Potential_benchmark_iron)).

## Workflow

To understand the following sections, it is important to know how the workflow is designed:

1. The user has to define a configuration file, its content will be illustrated in section [Configuration File Syntax](#configuration-file-syntax). The main idea is to set the parameters for eache phase of the pipeline, while also defining the requirements for each Slurm job.

2. Run the pipeline passing the configuration file, this step will create all the "watcher" jobs with their dependencies. A "watcher" job is a Slurm job that dispatches the real jobs for its phase, it is also used to run common processes and mark the end of a phase (when needed).

3. Without any additional flags, the pipeline will execute the following workflow:
    1. Find the best hyperparameters in the provided optimization space
    2. Get the best n models
    3. Train for additional epochs the best models
    4. Create the LAMMPS potentials using the best models
    5. Run the simulations using the obtained potentials

4. The results will be available in the defined sweep path using the following structure:

`sweep_path`
|---`hyper_search`
|   |---`loss_function_errors.csv` (summary of losses divided in energy and force)
|   |---`parameters.csv` (summary of loss and used parameters from the optimization space)
|   |---`1`
|   ...
|   |---`itern_n`
|       |---`1`
|       ...
|       |---`subiter_n`
|           |---`training_files`
|           |---`optimized_params.yaml` (parameters used for that subiteration)
|           |---`model_info.csv` (iter, subiter, loss, used for identification in the next phases)
|           |---`potential.in`(only if `--nodeep` is used)
|
|---`deep_train`
|   |---`1`
|   ...
|   |---`best_n`
|       |---`training_files`
|       |---`optimized_params.yaml`
|       |---`model_info.csv`
|       |---`potential.in`
|
|---`inference_bench`
|   |---`1`
|   ...
|   |---`best_n`
|       |---`bench_files`
|       |---`model_info.csv`
|       |---`timings.csv`
|
|---`properties_simulation`
    |---`1`
    ...
    |---`best_n`
        |---`simulation_files`
        |---`model_info.csv`
        |---`plots`

## Installation

To install the framework and its dependencies, follow these steps:

0. Ensure that your system has [Slurm](https://slurm.schedmd.com/documentation.html) installed and that your user has access to the commands `sbatch` and `squeue`. Currently only usage via Slurm is supported.

1. Create and activate a new Conda environment with the requirements of your choosen model. Currently supported models are:
    - [PACE](https://github.com/ICAMS/python-ace?tab=readme-ov-file) --> environment should be named `pl`
    - [MACE](https://github.com/ACEsuit/mace) (no properties simulation) --> environment should be named `mace`
    - [grACE](https://github.com/ICAMS/grace-tensorpotential) --> environment should be named `grace`

In the same environments install [simple_slurm](https://github.com/amq92/simple_slurm) and [XPOT](https://github.com/dft-dutoit/XPOT) from GIT. Do not use the PyPi versions.

2. Clone the repository and open it:
    ```bash
    git clone https://github.com/TheNewPing/POTline.git
    cd POTline
    ```

3. Install LAMMPS accordingly to the model's documentation. Ensure to enable GPU support.
You can also use the Slurm scripts under `src/configs/[cluster_name]/install_lammps/[model_name]` to start the installation after having completed the LAMMPS setup. Note that they are tailored for [Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius) or [Habrok](https://wiki.hpc.rug.nl/habrok/start), adjust them to your needs.
    - [PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/quickstart/#lammps)
    - [MACE](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html)
    - [grACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/#lammps-with-grace)

## Usage

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

The configuration file for POTline is written in HJSON format, which is a user-friendly extension of JSON. Some examples are provided in the folder `src/configs` (currently only Habrok configuration are updated), remember that when writing a configuration you have to keep in mind both the model and the cluster used.

Below is a description of the main sections and their respective parameters:

#### General
- `lammps_bin_path`: Path to the LAMMPS binary.
- `model_name`: Name of the model (currently supports `pacemaker, mace, gracemaker`).
- `best_n_models`: Number of best models to use in inference and simulation step.
- `hpc`: HPC mode, keep always True.
- `cluster`: Cluster configuration to use (currently supports only `snellius`).
- `sweep_path`: Output path for the experiments.
- `slurm_watcher`: Slurm options for general watcher, currently used only in the conversion phase.
- `slurm_opts`: Currently not used, keep always `{}`
- `modules`: Scripts to source for general jobs, currently used only in the conversion phase.
- `py_scripts`: Python scripts to run before general jobs execution, currently used only in the conversion phase.

### Deep training
- `max_epochs`: Max number of epochs for deeper training on best models.
- `slurm_watcher`: Slurm options for best models training watcher, has only to dispatch the training jobs and wait until they are all completed, so it needs a lot of time but low resources.
- `slurm_opts`: Slurm options for best models training jobs, allocate resources according to the model, GPU usage is reccomended.
- `modules`: Scripts to source for best models training.
- `py_scripts`: Python scripts to run before best models training.

#### Inference
- `prerun_steps`: Number of pre-run steps.
- `max_steps`: Maximum number of steps.
- `slurm_watcher`: Slurm options for inference watcher, has only to dispatch the inference jobs, currently no jobs deepend on this phase, so it requires low time and resources.
- `slurm_opts`: Slurm options for inference jobs, allocate resources according to the model, GPU usage is reccomended.
- `modules`: Scripts to source for inference.
- `py_scripts`: Python scripts to run before inference.

#### Data Analysis
- `lammps_inps_path`: Path to LAMMPS input files (from Potential_benchmark_iron).
- `pps_python_path`: Path to Python scripts for post-processing (from Potential_benchmark_iron).
- `ref_data_path`: Path to reference data (from Potential_benchmark_iron).
- `slurm_watcher`: Slurm options for simulation watcher, has only to dispatch the simulation jobs, currently no jobs deepend on this phase, so it requires low time and resources.
- `slurm_opts`: Slurm options for simulation jobs, allocate resources according to the model, GPU usage is reccomended.
- `modules`: Scripts to source for simulation.
- `py_scripts`: Python scripts to run before simulation.

#### Hyperparamerter optimization
- `max_iter`: Number of iterations of ask-tell for the baesyan optimizer.
- `n_initial_points`: Consult `skopt.Optimizer`.
- `n_points`: Number of parameters sets asked at each iteration to the optimizer.
- `strategy`: Strategy for the optimizer, consult `skopt.Optimizer`.
- `energy_weight`: Loss weight of the energy component (0.0 - 1.0).
- `slurm_watcher`: Slurm options for optimization watcher, used to dispatch the fitting jobs and to host the Bayesian optimizer. Requires "medium resources" and a lot of time. GPU is not needed.
- `slurm_opts`: Slurm options for optimization jobs, allocate resources according to the model, GPU usage is reccomended.
- `modules`: Scripts to source for optimization.
- `py_scripts`: Python scripts to run before optimization.
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
- Add cracks and dislocation properties simulations
- Add summary data presentation
- Test more model variants
- Test finetuning where available
