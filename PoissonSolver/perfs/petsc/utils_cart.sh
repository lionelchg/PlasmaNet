#!/bin/bash -l

# Generic function to solve the symmetrized Poisson matrix using printing in $1 filename
# using solver type $2, using preconditioner $3 with a tolerance of $4 an $5 number of processors
run_solver_cart() {
    for nn in 101 201 401 801 2001 4001 5001 5501 6001
    do
        mpirun -np "$1" ./src/poisson.out $nn $nn "$5"/$nn.log \
            -ksp_type "$2" -ksp_rtol "$4" -pc_type "$3"
    done
}

# Particular function for CG-GAMG
run_cg_gamg() {
    for nn in 101 201 401 801 2001 4001 5001 5501 6001
    do
        mpirun -np "$1" ./src/poisson.out $nn $nn "$3"/$nn.log -ksp_type cg \
            -ksp_rtol "$2" -pc_type gamg
    done
}

# Particular function for CG-HYPRE-BOOMERANG
run_cg_hypre_boomeramg() {
    for nn in 101 201 401 801 2001 4001 5001 5501 6001
    do
        mpirun -np "$1" ./src/poisson.out $nn $nn "$3"/$nn.log -ksp_type cg -ksp_rtol "$2" -pc_type hypre -pc_hypre_type boomeramg
    done
}

# Particular function for direct solver using lu/cholesky
run_direct_cart() {
    for nn in 5001 5501 6001
    do
        mpirun -np "$1" ./src/poisson.out $nn $nn "$3"/$nn.log -ksp_type preonly -pc_type "$2"
    done
}

# Particular function for CG-HYPRE-BOOMERANG
run_hypre_boomeramg() {
    for nn in 101 201 401 801 2001 4001 5001 5501 6001
    do
        mpirun -np "$1" ./src/poisson.out $nn $nn "$4"/$nn.log -ksp_type "$2" -ksp_rtol "$3" -pc_type hypre -pc_hypre_type boomeramg
    done
}