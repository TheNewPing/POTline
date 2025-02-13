"""
Metrics calculator module.
"""

from pathlib import Path
from math import sqrt
from typing import Tuple
import csv

import yaml

from ..loss_logger import INFO_FILENAME
from ..properties_simulator import PROPERTIES_BENCH_DIR_NAME
from ..inference_bencher import INFERENCE_BENCH_DIR_NAME

METRICS_DIR_NAME: str = 'metrics'
Q_FACTOR_REF_VALUES_NAME: str = 'q_factor.yaml'
SIM_RESULTS_DIR_NAME: str = 'data'
SIM_RESULTS_FILE_NAME: str = 'results.txt'
BENCH_RESULTS_FILE_NAME: str = 'bench_timings.csv'
Q_FACTOR_PATH: Path = Path(__file__).parent / Q_FACTOR_REF_VALUES_NAME

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

    def calculate_q_factors(self) -> dict[Tuple[int,int], float]:
        """
        Calculate the q-factors for the simulations.
        """
        # List of q-factors, each simulations is identified by a tuple of the iteration and subiteration
        q_factors: dict[Tuple[int,int], float] = {}

        # Load the reference values
        with Q_FACTOR_PATH.open('r') as file:
            ref_values = yaml.safe_load(file)

        sim_paths: list[Path] = [p for p in self._sim_path.iterdir() if p.is_dir()]
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

    def calculate_inference_time(self) -> dict[Tuple[int,int], float]:
        """
        Calculate the inference time for the simulations.
        """
        # List of inference times, each simulations is identified by a tuple of the iteration and subiteration
        inference_times: dict[Tuple[int,int], float] = {}

        inf_paths: list[Path] = [p for p in self._inf_path.iterdir() if p.is_dir()]
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
