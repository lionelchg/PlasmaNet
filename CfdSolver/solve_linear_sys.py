########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using CUDA                                              #
#                                                                                                                      #
#                                          Ekhi Ajuria , CERFACS, 20.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import torch
import plasmanet_cpp

def solveLinearSystemCG(pot,div, A_val, I_A, J_A):
    """Solves the linear system using the CG  method.
        Note: Since we don't receive a velocity field, we need to receive the is3D
        flag from the caller.

    Arguments:
        pot (Tensor): Tensor with domain size which will store solution.
        div (Vector): Rhs vector
        A_val (Vector): Diagonal values of the A matrix
        I_A (Vector): Csr format of A
        J_A (Vector): Csr format of A

    Output:
        p (Tensor): Pressure field
        p_tol: Maximum residual accross all batches.

    """
   
    h = pot.size(0)
    w = pot.size(1)
 
    assert pot.is_contiguous() and div.is_contiguous(), "Input is not contiguous"

    residual = 0.0
    p_tol = 1e-6
    max_iter = 1000
    plasmanet_cpp.solve_linear_system_CG(pot, div, A_val, I_A, J_A,p_tol,max_iter,residual)

    print("End sol sys, residual = ", residual)

    return residual
