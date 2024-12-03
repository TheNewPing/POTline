#!/bin/bash
#------------------------------
# This is a main/controlling code written in SHELL.
# Please read through README.md file carefully before using the script.
# DATE: 1 Dec 2021
# AUTHOR: lei.zhang@rug.nl
#-------------------------------
# 1st updates: 11 Oct 2022: New function
# Major change: all calculations are submitted by SLURM.
#-------------------------------
# 2nd updates: 2 Dec 2022 
# Major change: 
# - separate the lammps calculation and postprocessing
# - copy input sript and perform the calculation in one folder, 
# which allows multiple tests without replicating folders manually.
# - fix the inconsistency of Bain path calculation
# - Merge the surface energy input into one script
#------------------------------
# 3rd updates: 10 June 2023
# - add the calculation of T-S curve
# - add jace1x potential
#------------------------------

# Collect the input parameters
out_path=$1
hpc=$2
LMMP=$3
lmp_inps=$4
pps_python=$5
ref_data_path=$6
eaddress=$7

if [ "$hpc" = "True" ]; then
    LMMP="srun -G0 -n1 ${LMMP}"
    module load 2024
    module load Miniconda3/24.7.1-0
    module load 2022
    module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
    # Initialize Conda
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate pl
fi

# Change to the output directory
cd ${out_path}

# clear caches
rm dump*
rm *.csv
rm sfe*
rm -r ./data
rm -r ./plots
rm in.*
rm *.mod
rm *.py
rm results.txt
rm *.log

# create a data folder
mkdir data

#**********************************
# Get the information 
#**********************************
## locate the folder and grep the folder name
fullpath=${PWD}
potential_name=`echo $(basename $fullpath)`
# Grep the potential version and echo to results file
echo '#**********************************' | tee -a  ./data/results.txt
echo 'Potential basis set:' ${potential_name} | tee -a ./data/results.txt
awk '/^pair_style*/' ../potential.in | tee -a ./data/results.txt
awk '/^pair_coeff*/' ../potential.in | tee -a ./data/results.txt
echo '#**********************************' | tee -a ./data/results.txt

#**********************************
# Calculation section
#**********************************
# E-V curve 
cp ${lmp_inps}/in.eos .
mpirun -n 4 ${LMMP} -in in.eos -v folder ${potential_name}
# fit EOS
cp ${pps_python}/eos-fit.py .
python eos-fit.py
cp volume.dat ./data/eos_mlip.csv
# Get lattice parameter
a0=$(grep 'a0 =' ./data/results.txt | awk '{print $3}')

# Vacancy formation energy
cp ${lmp_inps}/in.vac .
${LMMP} -in in.vac -v lat ${a0}

# Calculation of elastic constants.--------------------------------
cp ${lmp_inps}/in.elastic .
cp ${lmp_inps}/*.mod .
${LMMP} -in in.elastic -v lat ${a0}

# Calculation of surface energies.---------------------------------
cp ${lmp_inps}/in.surf* .
# (100) plane
${LMMP} -in in.surf1 -v lat ${a0}
# (110) plane
${LMMP} -in in.surf2 -v lat ${a0}
# (111) plane
${LMMP} -in in.surf3 -v lat ${a0}
# (112) plane
${LMMP} -in in.surf4 -v lat ${a0}

# Bain path calculation.------------------------------------------
cp ${lmp_inps}/in.bain_path .
${LMMP} -in in.bain_path -v lat ${a0}
cp bain_path.csv ./data

# Stacking fault energy---------------------------------------------
cp ${lmp_inps}/in.sfe_* .
${LMMP} -in in.sfe_110 -v lat ${a0}
${LMMP} -in in.sfe_112 -v lat ${a0}
cp ./sfe_110.csv ./data
cp ./sfe_112.csv ./data

# Traction-separatio curve------------------------------------------
cp ${lmp_inps}/in.ts_* .
${LMMP} -in in.ts_100 -v lat ${a0}
${LMMP} -in in.ts_110 -v lat ${a0}
cp ./ts_100.csv ./data
cp ./ts_110.csv ./data

#**********************************
# Plotting section
#**********************************
# Execute python script to do the plots.py------------------------------
# Plot E-V curve and Bain path
cp -r ${ref_data_path} . 
mkdir plots
cd plots
cp ${pps_python}/eos_bain.py .
cp ${pps_python}/sfe.py .
cp ${pps_python}/ts.py .
python eos_bain.py
python sfe.py
python ts.py
rm *.py

echo "Finish plotting results!"

# delete all lammps inputs
cd ..
rm in.*
rm *.mod

# Send email of the plots as the attached file.
# Mail the results ---------------------------------------------------
if [ "$hpc" = "True" ]; then
    mail -s "Basic Properties of iron predicted by IAP" -a ./data/results.txt -a ./plots/eos_bp.png -a ./plots/sfe.png "${eaddress}" <<EOF
Please check the performance of interatomic potential: ${potential_name}
EOF
    echo "Mail the results successful!"
fi
