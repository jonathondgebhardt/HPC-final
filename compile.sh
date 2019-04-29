#!/usr/bin/env bash

# Compile source with mpicc
mpicc ${1}.c -o ${1}.mpi -lm

echo "Compiled source to ${1}.mpi"

# Edit batch file for slurm
sed  s/'NAME'/${1}/g ${HOME}/template.batch > ${1}.batch
sed -i s/'NODES'/${2}/g ${1}.batch
sed -i s/'TASKS'/${3}/g ${1}.batch

echo "Batch file is ${1}.batch"

