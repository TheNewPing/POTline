"""
Preset job slurm.
"""

from pathlib import Path
from enum import Enum

from ..dispatcher import SlurmCluster, JobType, SupportedModel
from .slurm_dispatcher import SlurmCluster

_file_path: Path = Path(__file__).parent.resolve()
_template_path: Path = _file_path / 'template'

class CommandsName(Enum):
    """
    Supported command names.
    """
    CONDA_PACE = 'conda_pace.sh'
    MOD_CONDA = 'module_conda.sh'
    MOD_CUDA = 'module_cuda.sh'
    MOD_MPI = 'module_mpi.sh'

def get_slurm_options(cluster: str, job_type: JobType, out_path: Path,# noqa: C901
                      model: SupportedModel | None = None,
                      n_cpu: int | None = None,
                      email: str | None = None) -> dict:
    """
    Get the SLURM options for the job.
    """
    if cluster == SlurmCluster.SNELLIUS.value:
        if job_type == JobType.INF:
            if not n_cpu:
                raise ValueError("n_cpu must be provided for inference jobs.")
            return {
                'job_name': f"inf_{str(model)}",
                'output': f"{str(out_path)}/inf_%j.out",
                'error': f"{str(out_path)}/inf_%j.err",
                'time': "03:00:00",
                'ntasks': 1,
                'cpus_per_task': n_cpu,
            }
        elif job_type == JobType.SIM:
            if not email:
                raise ValueError("Email must be provided for simulation jobs.")
            return {
                'job_name': f"sim_{str(model)}",
                'ntasks': 32,
                'cpus_per_task': 1,
                'mem': '10G',
                'time': '3:00:00',
                'error': f"{str(out_path)}/sim_%j.err",
                'output': f"{str(out_path)}/sim_%j.out",
                'mail_type': 'ALL',
                'mail_user': email,
            }
        elif job_type == JobType.FIT:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE:
                return {
                'job_name': f"fit_{str(model)}",
                'output': f"{str(out_path)}/fit_%j.out",
                'error': f"{str(out_path)}/fit_%j.err",
                'time': "12:00:00",
                'mem': "50G",
                'partition': "gpu",
                'nodes': 1,
                'ntasks': 1,
                'cpus_per_task': 16,
                'gpus': 1,
            }
            elif model == SupportedModel.MACE:
                raise NotImplementedError("MACE model not implemented.")
        elif job_type == JobType.DEEP:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE:
                return {
                    'job_name': f"deep_{str(model)}",
                    'output': f"{str(out_path)}/deep_%j.out",
                    'error': f"{str(out_path)}/deep_%j.err",
                    'time': "36:00:00",
                    'mem': "50G",
                    'partition': "gpu",
                    'nodes': 1,
                    'ntasks': 1,
                    'cpus_per_task': 16,
                    'gpus': 1,
                }
            elif model == SupportedModel.MACE:
                raise NotImplementedError("MACE model not implemented.")
        raise NotImplementedError(f"Job type {job_type} not implemented.")
    raise NotImplementedError(f"Cluster {cluster} not implemented.")


def get_slurm_commands(cluster: str, # noqa: C901
                       job_type: JobType,
                       model: SupportedModel | None = None) -> list[str]:
    """
    Get the commands for the job.
    """
    if cluster == SlurmCluster.SNELLIUS.value:
        if job_type == JobType.INF:
            return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}']
        elif job_type == JobType.SIM:
            return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}']
        elif job_type == JobType.FIT:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE:
                return [f'source {_template_path / cluster / CommandsName.MOD_CONDA.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_CUDA.value}',
                        f'source {_template_path / cluster / CommandsName.CONDA_PACE.value}']
            elif model == SupportedModel.MACE:
                raise NotImplementedError("MACE model not implemented.")
        elif job_type == JobType.DEEP:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE:
                return [f'source {_template_path / cluster / CommandsName.MOD_CONDA.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_CUDA.value}',
                        f'source {_template_path / cluster / CommandsName.CONDA_PACE.value}']
            elif model == SupportedModel.MACE:
                raise NotImplementedError("MACE model not implemented.")
        raise NotImplementedError(f"Job type {job_type} not implemented.")
    raise NotImplementedError(f"Cluster {cluster} not implemented.")
