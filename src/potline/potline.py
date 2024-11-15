import os

from .config_reader import ConfigReader
from .optimizer import Optimizer
from .lammps_runner import run_benchmark
from .lammps_analysis import run_properties_simulation

class PotLine():
    def __init__(self, config_path: str,
                 max_iter: int, 
                 with_inference: bool,
                 with_data_analysis: bool,):
        self.config_reader = ConfigReader(config_path)
        self.max_iter: int = max_iter
        self.optimizer: Optimizer = self.config_reader.create_optimizer()
        self.with_inference: bool = with_inference
        self.with_data_analysis: bool = with_data_analysis

    def run(self):
        self.optimizer.optimize(self.max_iter)
        self.optimizer.get_final_results()
        yace_path = self.optimizer.convert_yace(os.path.join(os.getcwd(), 'interim_potential_0.yaml'))
    
        if self.with_inference:
            run_benchmark()
        if self.with_data_analysis:
            run_properties_simulation()