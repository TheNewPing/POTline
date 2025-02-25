"""
Metrics calculator module.
"""

from pathlib import Path
from math import sqrt
from typing import Tuple, Any
import csv

import yaml
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error # type: ignore

from ..loss_logger import INFO_FILENAME
from ..properties_simulator import PROPERTIES_BENCH_DIR_NAME
from ..inference_bencher import INFERENCE_BENCH_DIR_NAME

METRICS_DIR_NAME: str = 'metrics'
Q_FACTOR_REF_VALUES_NAME: str = 'q_factor.yaml'
SIM_RESULTS_DIR_NAME: str = 'data'
SIM_RESULTS_FILE_NAME: str = 'results.txt'
BENCH_RESULTS_FILE_NAME: str = 'bench_timings.csv'
Q_FACTOR_PATH: Path = Path(__file__).parent / Q_FACTOR_REF_VALUES_NAME
SCREW_REF_PATH: Path = Path(__file__).parent / 'Meng_screw_dis.xyz'

class MetricsCalculator():
    """
    Class for running the metrics calculations.

    Args:
        sweep_path: The path to the sweep directory.
    """
    def __init__(self, sweep_path: Path):
        self._out_path = sweep_path / METRICS_DIR_NAME
        self._inf_path = sweep_path / INFERENCE_BENCH_DIR_NAME
        self._sim_path = sweep_path / PROPERTIES_BENCH_DIR_NAME

    def calculate_q_factors(self, run_nums: list[int] | None = None) -> dict[Tuple[int,int], float]:
        """
        Calculate the q-factors for the simulations.

        Args:
            run_nums: The list of run numbers to calculate the q-factors for.
            If None, all the simulations are used.

        Returns:
            A dictionary with the q-factors for each simulation.
            Uses a tuple of the iteration and subiteration as the key.
            (from the hyperparameter optimization step)
        """
        # List of q-factors, each simulations is identified by a tuple of the iteration and subiteration
        q_factors: dict[Tuple[int,int], float] = {}

        # Load the reference values
        with Q_FACTOR_PATH.open('r') as file:
            ref_values = yaml.safe_load(file)

        sim_paths: list[Path] = [p for p in self._sim_path.iterdir() if p.is_dir()]
        if run_nums: # Filter the simulations
            sim_paths = [p for p in sim_paths if int(p.name) in run_nums]
        for p in sim_paths:
            info_path = p / INFO_FILENAME
            # Load the iteration and subiteration
            with info_path.open('r') as file:
                data = yaml.safe_load(file)
                iteration = int(data['iteration'])
                subiteration = int(data['subiteration'])
            data_path = p / SIM_RESULTS_DIR_NAME / SIM_RESULTS_FILE_NAME
            # Load the calculated properties
            with data_path.open('r') as file:
                lines = file.readlines()
                properties = {
                    'a0': float(lines[11].split('=')[1].strip().split()[0]),  # Lattice Constant
                    'ev': float(lines[14]),                                   # Vacancy formation energy
                    'c11': float(lines[17].split('=')[1].strip().split()[0]), # Elastic Constant
                    'c12': float(lines[18].split('=')[1].strip().split()[0]), # Elastic Constant
                    'c44': float(lines[19].split('=')[1].strip().split()[0]), # Elastic Constant
                    'se100': float(lines[27]),                                # surface energy
                    'se110': float(lines[30]),                                # surface energy
                    'se111': float(lines[33]),                                # surface energy
                    'se112': float(lines[36])                                 # surface energy
                }

            # Calculate the q-factor
            norm_errors = {key: ((properties[key] - ref_values[key]) / ref_values[key]) ** 2
                           for key in ref_values}
            q_factor = sqrt(sum(norm_errors.values()) / len(norm_errors))
            q_factors[(iteration, subiteration)] = q_factor

        return q_factors

    def calculate_inference_time(self, run_nums: list[int] | None = None) -> dict[Tuple[int,int], float]:
        """
        Calculate the inference time for the simulations.

        Args:
            run_nums: The list of run numbers to calculate the inference times for.
            If None, all the simulations are used.

        Returns:
            A dictionary with the inference times for each simulation.
            Uses a tuple of the iteration and subiteration as the key.
            (from the hyperparameter optimization step)
        """
        # List of inference times, each simulations is identified by a tuple of the iteration and subiteration
        inference_times: dict[Tuple[int,int], float] = {}

        inf_paths: list[Path] = [p for p in self._inf_path.iterdir() if p.is_dir()]
        if run_nums: # Filter the simulations
            inf_paths = [p for p in inf_paths if int(p.name) in run_nums]
        for p in inf_paths:
            info_path = p / INFO_FILENAME
            # Load the iteration and subiteration
            with info_path.open('r') as file:
                data = yaml.safe_load(file)
                iteration = int(data['iteration'])
                subiteration = int(data['subiteration'])
            data_path = p / BENCH_RESULTS_FILE_NAME
            # Load the inference results
            with data_path.open('r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    inference_time = float(row['time_diff'])
                    prerun_steps = int(row['prerun_steps'])
                    max_steps = int(row['max_steps'])
                    inference_times[(iteration, subiteration)] = inference_time / (max_steps - prerun_steps)

        return inference_times

    def plot_screw_dislocation(self,
                               run_nums: list[int] | None = None) -> dict[str, list[dict[str, Any]]]:
        """
        Plot the screw dislocation.

        Args:
            run_nums: The list of run numbers to plot the screw dislocation for.
            If None, all the simulations are used.

        Returns:
            A dictionary with the metrics for each simulation
        """
        screw_in_atoms_dft = ase.io.read(SCREW_REF_PATH, ':')
        screw_ener_dft = np.array([at.get_potential_energy() for at in screw_in_atoms_dft])
        screw_dft_150at = screw_ener_dft[0:126]
        screw_dft_81at = screw_ener_dft[126:152]
        screw_dft_135at = screw_ener_dft[152::]
        h2spoints = [1,11,21,31,41,51,61,71,81,91,104,109,118]
        screw_dft_e2h_81 = 1000*(screw_dft_81at[0:13]-screw_dft_81at[0])
        screw_dft_h2s_81 = 1000*(screw_dft_81at[13::]-screw_dft_81at[0])
        screw_dft_h2s_150 = []
        for index in h2spoints:
            screw_dft_h2s_150.append(screw_dft_150at[index]*1000)
        screw_dft_h2s_150 = screw_dft_h2s_150 - screw_dft_h2s_150[0]

        sim_paths: list[Path] = [p for p in self._sim_path.iterdir() if p.is_dir()]
        if run_nums: # Filter the simulations
            sim_paths = [p for p in sim_paths if int(p.name) in run_nums]

        tot_out = {}

        for p in sim_paths:
            ener_ace = np.genfromtxt(p / 'data' / 'energy.dat')

            ace_150at = ener_ace[0:126]
            ace_81at = ener_ace[126:152]
            ace_135at = ener_ace[152::]

            ace_e2h_81 = 1000*(ace_81at[0:13]-ace_81at[0])
            ace_h2s_81 = 1000*(ace_81at[13::]-ace_81at[0])

            ace_h2s_150 = np.array([ace_150at[index]*1000 for index in h2spoints])
            ace_h2s_150 = ace_h2s_150 - ace_h2s_150[0]

            plt.rcParams.update({'font.size':18})
            plt.rcParams['figure.figsize'] = [20, 7]
            _, axs = plt.subplots(1, 3)

            # define a set of colors
            cc1='red' #'#F7AC08'
            cc4='#D55E00'

            axs[0].plot(np.arange(0,1.01,1/(len(screw_dft_e2h_81)-1)), screw_dft_e2h_81,
                        marker='o',markersize=12, c=cc1, label='DFT')
            axs[0].plot(np.arange(0,1.01,1/(len(ace_e2h_81)-1)),ace_e2h_81,
                        marker='d',markersize=12,c=cc4,label='PACE')

            axs[1].plot(np.arange(0,1.01,1/(len(screw_dft_h2s_81)-1)),screw_dft_h2s_81,
                        marker='o',markersize=12, c=cc1,label='81at, DFT')
            axs[1].plot(np.arange(0,1.01,1/(len(ace_h2s_81)-1)),ace_h2s_81,
                        marker='d',markersize=12, c=cc4,label='81at, ACE')

            axs[2].plot(np.arange(0,1.01,1/(len(screw_dft_h2s_150)-1)),np.array(screw_dft_h2s_150),
                        marker='o',markersize=12, c=cc1,label='150at, DFT')
            axs[2].plot(np.arange(0,1.01,1/(len(ace_h2s_150)-1)),ace_h2s_150,
                        marker='d',markersize=12, c=cc4,label='150at, ACE')

            axs[0].set_ylim(-20,150)
            axs[1].set_ylim(-20,150)
            axs[2].set_ylim(-40,60)

            axs[0].legend(loc=(0.8,1.05), ncols=4, markerscale=1.5, fontsize=20)
            axs[0].set_ylabel('Energy / (meV/b)')
            axs[0].set_xlabel('Reaction Coords.')
            axs[1].set_xlabel('Reaction Coords.')
            axs[2].set_xlabel('Reaction Coords.')

            axs[0].annotate("(a)", xy=(-0.17, 1), weight='bold', xycoords="axes fraction", fontsize=26)
            axs[1].annotate("(b)", xy=(-0.17, 1), weight='bold', xycoords="axes fraction", fontsize=26)
            axs[2].annotate("(c)", xy=(-0.17, 1), weight='bold', xycoords="axes fraction", fontsize=26)

            plt.savefig(p / 'plots' / 'screw_disloc.png', format='png', dpi=600)

            p_out = []

            p_out.append({
                "property": "Screw e2h 81",
                "mae": mean_absolute_error(screw_dft_e2h_81, ace_e2h_81),
                "rmse": np.sqrt(mean_squared_error(screw_dft_e2h_81, ace_e2h_81)),
                'delta': ace_e2h_81[-1] - ace_e2h_81[0]
            })
            p_out.append({
                "property": "Screw h2s 81",
                "mae": mean_absolute_error(screw_dft_h2s_81, ace_h2s_81),
                "rmse": np.sqrt(mean_squared_error(screw_dft_h2s_81, ace_h2s_81)),
                'delta': ace_h2s_81[-1] - ace_h2s_81[0]
            })
            p_out.append({
                "property": "Screw h2s 150",
                "mae": mean_absolute_error(screw_dft_h2s_150, ace_h2s_150),
                "rmse": np.sqrt(mean_squared_error(screw_dft_h2s_150, ace_h2s_150)),
                'delta': ace_h2s_150[-1] - ace_h2s_150[0]
            })

            plt.rcParams.update({'font.size':20})
            plt.rcParams['figure.figsize'] = [10, 10]
            _, axs = plt.subplots(1, 1)

            #axs.set_title('Dipole configs. at finite T')
            axs.plot(np.arange(len(screw_dft_135at)), 1000*(screw_dft_135at-min(screw_dft_135at))/2,
                        c='black', label='DFT')

            axs.plot(np.arange(len(ace_135at)), 1000*(ace_135at-min(ace_135at))/2, c='blue',label='ML-IAP')

            axs.set_ylim(0,4000)
            axs.set_xlim(0,152)

            axs.legend(loc=(0,1.05), ncols=4, markerscale=1.5, fontsize=20)
            axs.set_ylabel('Energy, (meV/b)')
            axs.set_xlabel('Configuration No.')

            plt.savefig(p / 'plots' / 'screw_config.png', format='png', dpi=600)

            p_out.append({
                "property": "Screw Config Energy",
                "mae": mean_absolute_error(1000*(screw_dft_135at-min(screw_dft_135at))/2,
                                           1000*(ace_135at-min(ace_135at))/2),
                "rmse": np.sqrt(mean_squared_error(1000*(screw_dft_135at-min(screw_dft_135at))/2,
                                                   1000*(ace_135at-min(ace_135at))/2))
            })

            tot_out[p.name] = p_out

        return tot_out
