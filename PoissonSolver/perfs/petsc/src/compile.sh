#!/bin/bash
if [ $AVBP_HOSTTYPE == "MACBOOK" ]; then
   export PETSC_DIR=/usr/local/Cellar/petsc/3.15.4/

   mpif90 -cpp -g -Wall -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main.f90 -o poisson.out
   mpif90 -cpp -g -Wall -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main_axi.f90 -o poisson_axi.out

elif [ $AVBP_HOSTTYPE == "KRAKEN" ]; then
   export PETSC_DIR=/softs/local_intel/petsc/3.13.4
   export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH

   mpiifort -cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main.f90 -o poisson.out
   mpiifort -cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main_axi.f90 -o poisson_axi.out

elif [ $AVBP_HOSTTYPE == "NEMO" ]; then
   export PETSC_DIR=/data/softs/local_intel18/petsc/3.13.4/
   export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH

   mpiifort -cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main.f90 -o poisson.out
   mpiifort -cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc mod_linsystem.f90 main_axi.f90 -o poisson_axi.out

else
   echo "Undefined or invalid AVBP_HOSTTYPE..."
fi

