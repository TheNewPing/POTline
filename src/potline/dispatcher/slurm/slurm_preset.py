"""
Preset job slurm.
"""

from pathlib import Path
from enum import Enum

from ..dispatcher import SlurmCluster, JobType, SupportedModel
from ...hyper_searcher import OPTIM_DIR_NAME
from ...deep_trainer import DEEP_TRAIN_DIR_NAME
from ...inference_bencher import INFERENCE_BENCH_DIR_NAME
from ...properties_simulator import PROPERTIES_BENCH_DIR_NAME

_file_path: Path = Path(__file__).parent.resolve()
_template_path: Path = _file_path / 'template'

_faulty_nodes = 'gcn25,gcn56'

_job_dirs = {
    JobType.FIT: OPTIM_DIR_NAME,
    JobType.INF: INFERENCE_BENCH_DIR_NAME,
    JobType.DEEP: DEEP_TRAIN_DIR_NAME,
    JobType.SIM: PROPERTIES_BENCH_DIR_NAME,
}

class CommandsName(Enum):
    """
    Supported command names.
    """
    TF_GPU_TEST = 'tf_gpu_test.py'
    PYT_GPU_TEST = 'pyt_gpu_test.py'
    CONDA_PACE = 'conda_pace.sh'
    CONDA_MACE = 'conda_mace.sh'
    CONDA_GRACE = 'conda_grace.sh'
    MOD_MPI = 'module_mpi.sh'
    MOD_SIM = 'module_prop_sim.sh'
    MOD_MKL = 'module_mkl.sh'

def make_base_options(job: JobType, model: SupportedModel, out_path: Path,
                      time: str, mem: str, ntasks: int, cpus_per_task: int) -> dict:
    """
    Make the base options for the job.
    """
    return {
        'job_name': f"{job.value}_{str(model)}",
        'output': f"{str(out_path)}/{job.value}_%j.out",
        'error': f"{str(out_path)}/{job.value}_%j.err",
        'time': time,
        'mem': mem,
        'nodes': 1,
        'ntasks': ntasks,
        'cpus_per_task': cpus_per_task,
        'requeue': True,
    }

def make_gpu_array_options(job: JobType, model: SupportedModel, out_path: Path,
                    time: str, mem: str, ntasks: int, cpus_per_task: int, gpus: int,
                    partition: str, array_ids: list[int]) -> dict:
    """
    Make the GPU options for the job.
    """
    out_job_path: str = f"{str(out_path)}/%a/{_job_dirs[job]}"
    return {
        **make_base_options(job, model, out_path, time, mem, ntasks, cpus_per_task),
        'chdir': out_job_path,
        'output': f"{out_job_path}/{job.value}_%A_%a.out",
        'error': f"{out_job_path}/{job.value}_%A_%a.err",
        'gpus': gpus,
        'partition': partition,
        'exclude':_faulty_nodes,
        'array': array_ids,
    }

def get_slurm_options(cluster: str, job_type: str, out_path: Path, # noqa: C901
                      model: str | None = None,
                      n_cpu: int | None = None,
                      email: str | None = None,
                      array_ids: list[int] | None = None) -> dict:
    """
    Get the SLURM options for the job.
    """
    if cluster == SlurmCluster.SNELLIUS.value:
        if job_type == JobType.INF.value:
            if not n_cpu:
                raise ValueError("n_cpu must be provided for inference jobs.")
            return make_gpu_array_options(
                JobType.INF, model, out_path, "3:00:00", "50G", 1, n_cpu, 1, "gpu_a100", array_ids)
        elif job_type == JobType.SIM.value:
            if not email:
                raise ValueError("Email must be provided for simulation jobs.")
            return {
                **make_gpu_array_options(
                    JobType.SIM, model, out_path, "3:00:00", "50G", 1, n_cpu, 1, "gpu_a100", array_ids),
                'mail_type': 'ALL',
                'mail_user': email,
            }
        elif job_type == JobType.FIT.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return make_gpu_array_options(
                    JobType.FIT, model, out_path, "12:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            elif model == SupportedModel.MACE.value:
                return make_gpu_array_options(
                    JobType.FIT, model, out_path, "12:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            elif model == SupportedModel.GRACE.value:
                return make_gpu_array_options(
                    JobType.FIT, model, out_path, "12:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.DEEP.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return make_gpu_array_options(
                    JobType.DEEP, model, out_path, "36:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            elif model == SupportedModel.MACE.value:
                return make_gpu_array_options(
                    JobType.DEEP, model, out_path, "36:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            elif model == SupportedModel.GRACE.value:
                return make_gpu_array_options(
                    JobType.DEEP, model, out_path, "36:00:00", "50G", 1, 16, 1, "gpu_a100", array_ids)
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.MAIN.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return make_base_options(JobType.MAIN, model, out_path, "119:00:00", "50G", 1, 16)
            elif model == SupportedModel.MACE.value:
                return make_base_options(JobType.MAIN, model, out_path, "119:00:00", "50G", 1, 16)
            elif model == SupportedModel.GRACE.value:
                return make_base_options(JobType.MAIN, model, out_path, "119:00:00", "50G", 1, 16)
            raise NotImplementedError(f"Model {model} not implemented.")
        raise NotImplementedError(f"Job type {job_type} not implemented.")
    raise NotImplementedError(f"Cluster {cluster} not implemented.")


def get_slurm_commands(cluster: str, # noqa: C901
                       job_type: str,
                       model: str | None = None) -> list[str]:
    """
    Get the commands for the job.
    """
    if cluster == SlurmCluster.SNELLIUS.value:
        if job_type == JobType.INF.value:
            if model is None:
                raise ValueError("Model must be provided for inference jobs.")
            if model == SupportedModel.PACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}']
            elif model == SupportedModel.MACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_MKL.value}']
            elif model == SupportedModel.GRACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}']
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.SIM.value:
            if model is None:
                raise ValueError("Model must be provided for simulation jobs.")
            if model == SupportedModel.PACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_SIM.value}']
            elif model == SupportedModel.MACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_MKL.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_SIM.value}']
            elif model == SupportedModel.GRACE.value:
                return [f'source {_template_path / cluster / CommandsName.MOD_MPI.value}',
                        f'source {_template_path / cluster / CommandsName.MOD_SIM.value}']
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.FIT.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_PACE.value}',
                        f'python {_template_path / CommandsName.TF_GPU_TEST.value}']
            elif model == SupportedModel.MACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_MACE.value}',
                        f'python {_template_path / CommandsName.PYT_GPU_TEST.value}']
            elif model == SupportedModel.GRACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_GRACE.value}',
                        f'python {_template_path / CommandsName.TF_GPU_TEST.value}']
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.DEEP.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_PACE.value}',
                        f'python {_template_path / CommandsName.TF_GPU_TEST.value}']
            elif model == SupportedModel.MACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_MACE.value}',
                        f'python {_template_path / CommandsName.PYT_GPU_TEST.value}']
            elif model == SupportedModel.GRACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_GRACE.value}',
                        f'python {_template_path / CommandsName.TF_GPU_TEST.value}']
            raise NotImplementedError(f"Model {model} not implemented.")
        elif job_type == JobType.MAIN.value:
            if model is None:
                raise ValueError("Model must be provided for fitting jobs.")
            if model == SupportedModel.PACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_PACE.value}',]
            elif model == SupportedModel.MACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_MACE.value}',]
            elif model == SupportedModel.GRACE.value:
                return [f'source {_template_path / cluster / CommandsName.CONDA_GRACE.value}',]
            raise NotImplementedError(f"Model {model} not implemented.")
        raise NotImplementedError(f"Job type {job_type} not implemented.")
    raise NotImplementedError(f"Cluster {cluster} not implemented.")
