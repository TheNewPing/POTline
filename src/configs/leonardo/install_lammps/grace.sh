#!/bin/bash
#SBATCH --job-name=lmp_inst
#SBATCH --output=lmp_inst_%j.out
#SBATCH --error=lmp_inst_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

module load nvhpc/24.3
module load hpcx/2.18.1--binary
module load openmpi/4.1.6--nvhpc--24.3
module load cmake/3.27.7

# module load gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0 bzip2/1.0.8--gcc--12.2.0-cc63fjd libmd/1.0.4--gcc--12.2.0-6jsw2r6 libbsd/0.11.7--gcc--12.2.0-liwo4uq expat/2.5.0--gcc--12.2.0-ldskudz ncurses/6.4--gcc--12.2.0-faagir2 readline/8.2--gcc--12.2.0-ex2p6kn gdbm/1.23--gcc--12.2.0-lghxvdl libiconv/1.17--gcc--12.2.0-s6uxr4v xz/5.4.1--gcc--12.2.0-up42vbf zlib/1.3--gcc--12.2.0 libxml2/2.10.3--gcc--12.2.0-7hnyzym pigz/2.7--gcc--12.2.0-7httwmy zstd/1.5.5--gcc--12.2.0-h6hxgmk tar/1.34--gcc--12.2.0-fuernar gettext/0.22.3--gcc--12.2.0-rtv3jgn libffi/3.4.4--gcc--12.2.0-6onkavz libxcrypt/4.4.35--gcc--12.2.0-zmvrvrk sqlite/3.43.2--gcc--12.2.0-gu5hlbk util-linux-uuid/2.38.1--gcc--12.2.0-jk5pswi python/3.11.6--gcc--12.2.0-nlkgjki
# module load py-mpi4py/3.1.4--openmpi--4.1.6--gcc--12.2.0

# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate grace
source ~/repos/POTline/src/configs/leonardo/modules/conda_grace.sh 

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
      -C ../cmake/presets/nvhpc.cmake \
      ../cmake
cmake --build . -- -j 32