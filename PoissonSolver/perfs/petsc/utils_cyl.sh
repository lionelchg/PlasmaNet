#!/bin/bash -l

# Generic function to solve the symmetrized Poisson matrix using printing in $1 filename
# using solver type $2, using preconditioner $3 with a tolerance of $4 an $5 number of processors
run_solver_cyl() {
    nnxs=(401 801 1601)
    nnys=(101 201 401)
    for nn in 101 201 401 801 2001 4001
    do
        mpirun -np "$1" ./src/poisson_axi.out $nn $nn "$5"/$nn.log -ksp_type "$2" -ksp_rtol "$4" -pc_type "$3"
    done
}

# Particular function for CG-GAMG
run_cg_gamg() {
    for nn in 101 201 401 801 2001 4001
    do
        mpirun -np "$1" ./src/poisson_axi.out $nn $nn "$3"/solver_cg_gamg_"$1"_procs_$nn.log -ksp_type cg -ksp_rtol "$2" -pc_type gamg
    done
}

# Particular function for CG-HYPRE-BOOMERANG
run_cg_hypre_boomerang() {
    for nn in 101 201 401 801 2001 4001
    do
        mpirun -np "$1" ./src/poisson_axi.out $nn $nn "$3"/solver_cg_hypre_boomerang_"$1"_procs_$nn.log -ksp_type cg -ksp_rtol "$2" -pc_type hypre -pc_hypre_type boomeramg
    done
}

# Particular function for direct solver using lu/cholesky
run_direct_cyl() {
    for nn in 101 201 401 801 2001 4001
    do
        mpirun -np "$1" ./src/poisson_axi.out $nn $nn "$3"/solver_lu_"$1"_procs_$nn.log -ksp_type preonly -pc_type "$2"
    done
}

