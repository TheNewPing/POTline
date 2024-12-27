#!/bin/bash
#SBATCH --job-name=lmp_grace_inst
#SBATCH --output=lmp_grace_inst_%j.out
#SBATCH --error=lmp_grace_inst_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate grace

cd lammps_versions
cd grace

if [ ! -d "lammps" ]; then
      git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git
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
      -D PKG_MC=ON \
      ../cmake
cmake --build . -- -j 32