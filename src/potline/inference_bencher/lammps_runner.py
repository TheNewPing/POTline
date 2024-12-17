"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path

from ..config_reader import BenchConfig
from ..dispatcher import Dispatcher, DispatcherManager
from ..model import PotModel

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME

_N_CPU: int = 1

def run_benchmark(model: PotModel, config: BenchConfig,
                  dispatcher_manager: DispatcherManager):
    """
    Run the LAMMPS benchmark.

    Args:
        - model: model to benchmark
        - config: benchmark configuration
        - dispatcher_manager: manager for dispatching the benchmark
    """
    inf_bench_dir: Path = model.get_out_path().parent / INFERENCE_BENCH_DIR_NAME
    inf_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in [
        'bash', BENCH_SCRIPT_TEMPLATE_PATH, _N_CPU,
        f'"{config.lammps_bin_path} {model.get_lammps_params()}"',
        LAMMPS_IN_PATH, config.prerun_steps,
        config.max_steps, inf_bench_dir]]

    dispatcher: Dispatcher = dispatcher_manager.create_dispatcher(
        [' '.join(command)],inf_bench_dir, model=model.get_name().value, n_cpu=_N_CPU)
    dispatcher.dispatch()
