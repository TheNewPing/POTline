"""
Properties simulation.
"""

from pathlib import Path
import shutil

from ..config_reader import ConfigReader
from ..loss_logger import ModelTracker

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

class PropertiesSimulator():
    """
    Class for running the LAMMPS properties simulations.

    Args:
        - config: configuration for the simulations
        - tracker_list: trackers to simulate
    """

    LAMMPS_INPS_PATH: Path = Path(__file__).parent / 'pot_testing' / 'lmps_inputs'
    PPS_PYTHON_PATH: Path = Path(__file__).parent / 'pot_testing' / 'py_pps'
    REF_DATA_PATH: Path = Path(__file__).parent / 'pot_testing' / 'REF_DATA'

    def __init__(self, config_path: Path, tracker_list: list[ModelTracker]):
        self._config = ConfigReader(config_path).get_prop_config()
        self._tracker_list = tracker_list
        self._out_path = self._config.sweep_path / PROPERTIES_BENCH_DIR_NAME

    def prep_sim(self) -> None:
        """
        Prepare the simulation directories.
        """
        self._out_path.mkdir(exist_ok=True)

        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i+1)
            iter_path.mkdir(exist_ok=True)
            shutil.copy(SUBMIT_TEMPLATE_PATH, iter_path)
            shutil.copy(tracker.model.get_pot_path(), iter_path)
            tracker.save_info(iter_path)
