"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path

from ..config_reader import BenchConfig
from ..dispatcher import Dispatcher, DispatcherFactory

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME

def run_benchmark(fitted_path: Path, config: BenchConfig,
                  dispatcher_factory: DispatcherFactory):
    """
    Run the LAMMPS benchmark.

    Args:
        - fitted_path: path to the directory with the fitted models.
        - config: configuration for the inference benchmark.
    """
    inf_bench_dir: Path = fitted_path / INFERENCE_BENCH_DIR_NAME
    inf_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in ['bash', BENCH_SCRIPT_TEMPLATE_PATH, config.n_cpu,
                                               config.lammps_bin_path, LAMMPS_IN_PATH, config.prerun_steps,
                                               config.max_steps, inf_bench_dir]]

    dispatcher: Dispatcher = dispatcher_factory.create_dispatcher(command, inf_bench_dir)
    dispatcher.dispatch()
