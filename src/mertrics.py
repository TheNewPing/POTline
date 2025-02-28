"""
This script contains the calculation for additional metrics about the experiments.
If you are using this repository for new experiments, use the following content only as an example.
"""

from pathlib import Path
from potline.metrics_builder import MetricsCalculator

grace_base = MetricsCalculator(Path('/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/full_test/grace'))
print('GRACE Base')
print(grace_base.calculate_q_factors())

grace_2l_omat_metrics = MetricsCalculator(Path(
    '/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/foundation/GRACE-2L-OAM_28Jan25'))
print('GRACE 2L OMAT')
print(grace_2l_omat_metrics.calculate_q_factors())

grace_1l_omat_metrics = MetricsCalculator(Path(
    '/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/foundation/GRACE-1L-OAM_2Feb25'))
print('GRACE 1L OMAT')
print(grace_1l_omat_metrics.calculate_q_factors())

grace_1l_mp_metrics = MetricsCalculator(Path(
    '/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/foundation/MP_GRACE_1L_r6_07Nov2024'))
print('GRACE 1L MP')
print(grace_1l_mp_metrics.calculate_q_factors())

grace_2l_r5_metrics = MetricsCalculator(Path(
    '/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/foundation/MP_GRACE_2L_r5_07Nov2024'))
print('GRACE 2L R5 MP')
print(grace_2l_r5_metrics.calculate_q_factors())

grace_2l_r6_metrics = MetricsCalculator(Path(
    '/leonardo_work/IscrC_GNNAIron/erodaro0/POTline/foundation/MP_GRACE_2L_r6_11Nov2024'))
print('GRACE 2L R6 MP')
print(grace_2l_r6_metrics.calculate_q_factors())
