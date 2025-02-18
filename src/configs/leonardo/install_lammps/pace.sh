#!/bin/bash
#SBATCH --job-name=lmp_pace_inst
#SBATCH --output=lmp_pace_inst_%j.out
#SBATCH --error=lmp_pace_inst_%j.err
#SBATCH --account=IscrC_GNNAIron
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

module load nvhpc/24.3
module load hpcx/2.18.1--binary
module load openmpi/4.1.6--nvhpc--24.3

cd lammps_versions
cd pace

if [ ! -d "lammps" ]; then
      git clone https://github.com/lammps/lammps.git
fi

cd lammps

if [ -d "build" ]; then
      rm -rf build
fi

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D BUILD_MPI=ON \
      -D PKG_ML-PACE=ON \
      -C ../cmake/presets/nvhpc.cmake \
      ../cmake
cmake --build . -- -j 32
