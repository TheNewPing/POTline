import hjson # type: ignore

from ..optimizer import Optimizer
from ..optimizer import XpotAdapter

class ConfigReader():
    """
    Class for reading and converting configuration files.
    A config file is written in hjson format and should have the following main sections:
    - optimizer: contains the configuration for the optimizer, uses the XPOT format.
    - inference: contains the configuration for the inference benchmark with LAMMPS.
    - data_analysis: contains the configuration for the data analysis on mechanical properties with LAMMPS.
    """
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        with open(file_path, 'r') as file:
            self.config_data: dict = hjson.load(file)
    
    def create_optimizer(self) -> Optimizer:
        converted_file_path: str = self.file_path.replace('.hjson', '_converted.hjson')
        if 'optimizer' not in self.config_data:
            raise ValueError('No optimizer configuration found in the config file.')
        if 'xpot' not in self.config_data['optimizer']:
            raise ValueError('No XPOT configuration found in the optimizer configuration.')
        with open(converted_file_path, 'w') as converted_file:
            hjson.dump(self.config_data['optimizer'], converted_file)
        return XpotAdapter(converted_file_path)
    
    def get_inf_benchmark_config(self) -> dict:
        if 'inference' not in self.config_data:
            raise ValueError('No inference configuration found in the config file.')
        return self.config_data['inference']
    
    def get_data_analysis_config(self) -> dict:
        if 'data_analysis' not in self.config_data:
            raise ValueError('No data analysis configuration found in the config file.')
        return self.config_data['data_analysis']