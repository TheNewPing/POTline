#!/bin/bash
#SBATCH --job-name=lmp_mace_inst
#SBATCH --output=lmp_mace_inst_%j.out
#SBATCH --error=lmp_mace_inst_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0
module load imkl/2024.2.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace

cd lammps_versions
cd mace

if [ ! -d "libtorch-shared-with-deps-1.13.0+cpu.zip" ]; then
      wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip
fi

if [ ! -d "libtorch" ]; then
      unzip libtorch-shared-with-deps-1.13.0+cpu.zip
fi

if [ ! -d "lammps" ]; then
      git clone --branch mace --depth=1 https://github.com/ACEsuit/lammps
fi

cd lammps

if [ -d "build" ]; then
      rm -rf build
fi

mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=$(pwd) \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_STANDARD_REQUIRED=ON \
      -D BUILD_MPI=ON \
      -D BUILD_OMP=ON \
      -D PKG_OPENMP=ON \
      -D PKG_ML-MACE=ON \
      -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch \
      ../cmake
cmake --build . -- -j 32
