module mod_linsystem
#include <petsc/finclude/petscksp.h>
use petscksp
use iso_c_binding

implicit none

integer, parameter :: len_str = 100
integer :: len_arg

type linsystem
    ! Global size of the matrix
    PetscInt :: global_size

    ! Local size of the matrix with beginning and end of row numbers of the proc
    PetscInt :: local_size, istart, iend

    ! Rank and number of procs
    PetscInt :: myrank, nprocs

    ! Linear system variables
    Mat :: A
    Vec :: sol, rhs
    KSP :: solver
    PC :: precond

    ! Pointers to the vectors for interaction with fortran arrays
    PetscScalar, dimension(:), pointer :: rhs_pointer, sol_pointer

    ! Casename
    character(len=len_str) :: casename

contains
    procedure :: initSystem
    procedure :: assembleMatrix
    procedure :: initSolver
    procedure :: solverInfo
    procedure :: vectorsInfo
    procedure :: matrixInfo
    procedure :: solve
    procedure :: solveInfo
    procedure :: dealloc
end type linsystem

contains
    subroutine initSystem(this, global_size, casename)
        PetscInt, intent(in) :: global_size
        character(len=len_str) :: casename
        class(linsystem) :: this
        PetscErrorCode :: ierr

        ! Name of the case
        this%casename = casename

        ! Assignement of global size of matrix
        this%global_size = global_size

        ! Start of PETSc
        call PetscInitialize(PETSC_NULL_CHARACTER, ierr)

        ! Initialization of the matrix
        call MatCreate(PETSC_COMM_WORLD, this%A, ierr)
        call MatSetType(this%A, MATMPIAIJ, ierr)
        call MatSetSizes(this%A, PETSC_DECIDE, PETSC_DECIDE, this%global_size, this%global_size, ierr)

        ! Proper preallocation memory
        call MatMPIAIJSetPreallocation(this%A, 5, PETSC_NULL_INTEGER, 4, PETSC_NULL_INTEGER, ierr)

        call MPI_Comm_rank(PETSC_COMM_WORLD, this%myrank, ierr)
        call MPI_Comm_size(PETSC_COMM_WORLD, this%nprocs, ierr)

        ! Fill the matrix with 5 diagonals (4, -1, -1, -1, -1)
        call MatGetOwnershipRange(this%A, this%istart, this%iend, ierr)
        this%local_size = this%iend - this%istart

    end subroutine initSystem

    subroutine assembleMatrix(this, indices, values)
        class(linsystem) :: this
        PetscInt :: i, global_index
        PetscInt, dimension(this%local_size, 5) :: indices
        PetscScalar, dimension(this%local_size, 5) :: values
        PetscErrorCode :: ierr

        ! Method to fill the matrix with merge(value1, value2, cond)
        do i = 1, this%local_size
            global_index = i + this%istart - 1
            call MatSetValues(this%A, 1, global_index, 5, indices(i, :), values(i, :), INSERT_VALUES, ierr)
        end do

        call MatAssemblyBegin(this%A, MAT_FINAL_ASSEMBLY, ierr)
        call MatAssemblyEnd(this%A, MAT_FINAL_ASSEMBLY, ierr)

        ! Creates the two vectors rhs and sol
        call MatCreateVecs(this%A, this%rhs, this%sol, ierr)

    end subroutine

    subroutine initSolver(this)
        class(linsystem) :: this
        PetscErrorCode :: ierr
        PetscScalar :: rtol = 1.0e-10, atol = 1.0e-20, dtol = 1.0e3
        PetscInt :: maxits = 5000

        call KSPCreate(PETSC_COMM_WORLD, this%solver, ierr)
        call KSPSetOperators(this%solver, this%A, this%A, ierr)
        call KSPSetType(this%solver, KSPCG, ierr)
        call KSPSetTolerances(this%solver, rtol, atol, dtol, maxits, ierr)
        call KSPSetFromOptions(this%solver, ierr)
        call KSPSetUp(this%solver, ierr)

    end subroutine initSolver

    subroutine vectorsInfo(this)
        ! Print the matrix
        class(linsystem) :: this
        PetscErrorCode :: ierr
        PetscViewer :: viewer

        ! Matrices to vectors.m file
        call PetscViewerASCIIOpen(PETSC_COMM_WORLD, "vectors.m", viewer, ierr)
        call PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INDEX, ierr)
        call VecView(this%rhs, viewer, ierr)
        call VecView(this%sol, viewer, ierr)
        call PetscViewerPopFormat(viewer, ierr)
        call PetscViewerDestroy(viewer, ierr)
    end subroutine vectorsInfo

    subroutine matrixInfo(this)
        ! Print the matrix
        class(linsystem) :: this
        PetscErrorCode :: ierr
        PetscViewer :: viewer

        ! Matrices to matrix.m file
        call PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrix.m", viewer, ierr)
        call PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT, ierr)
        call MatView(this%A, viewer, ierr)
        call PetscViewerPopFormat(viewer, ierr)
        call PetscViewerDestroy(viewer, ierr)
    end subroutine matrixInfo

    subroutine solverInfo(this)
        ! Give solver information (KSPType, PCType)
        class(linsystem) :: this
        PetscErrorCode :: ierr
        PetscViewer :: viewer

        call KSPView(this%solver, PETSC_VIEWER_STDOUT_WORLD, ierr)
        ! Print solver info to file
        call PetscViewerASCIIOpen(PETSC_COMM_WORLD, this%casename, viewer, ierr)
        call PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT, ierr)
        call KSPView(this%solver, viewer, ierr)
        call PetscViewerPopFormat(viewer, ierr)
        call PetscViewerDestroy(viewer, ierr)
    end subroutine solverInfo

    subroutine solve(this, rhs, sol)
        ! solve the linear system A * sol = rhs
        class(linsystem) :: this
        double precision, dimension(this%local_size) :: rhs
        double precision, dimension(this%local_size) :: sol
        integer :: i, global_index
        PetscErrorCode :: ierr

        ! Transfer rhs values into rhs_pointer (pointer to PetscVec)
        call VecGetArrayF90(this%rhs, this%rhs_pointer, ierr)
        do i = 1, this%local_size
            global_index = i + this%istart - 1
            this%rhs_pointer(i) = rhs(i)
        end do
        call VecRestoreArrayF90(this%rhs, this%rhs_pointer, ierr)

        ! Solve linear system
        call KSPSolve(this%solver, this%rhs, this%sol, ierr)

        ! Transfer Vec sol values into fortran array sol (pointer to PetscVec)
        call VecGetArrayF90(this%sol, this%sol_pointer, ierr)
        do i = 1, this%local_size
            sol(i) = this%sol_pointer(i)
            global_index = i + this%istart - 1
        end do
        call VecRestoreArrayF90(this%sol, this%sol_pointer, ierr)

    end subroutine solve

    subroutine solveInfo(this)
        ! Give information about the last use of solver
        class(linsystem) :: this
        PetscInt :: reason, its
        PetscScalar :: rnorm
        PetscErrorCode :: ierr

        call KSPGetConvergedReason(this%solver, reason, ierr)
        call KSPGetIterationNumber(this%solver, its, ierr)
        call KSPGetResidualNorm(this%solver, rnorm, ierr)

        if (this%myrank == 0) then
            open(11, file=this%casename, status="old", position="append", action="write")
            write(11, *) '-------------------------------'
            write(11, "('Converged reason:  ', I0)") reason
            write(11, "('Iterations number: ', I0)") its
            write(11, "('Residual norm:     ', ES10.3)") rnorm
            close(11)
        end if

    end subroutine solveInfo

    subroutine dealloc(this)
        ! Destroy all the created vectors and matrices
        class(linsystem) :: this
        PetscErrorCode :: ierr

        call MatDestroy(this%A, ierr)
        call VecDestroy(this%sol, ierr)
        call VecDestroy(this%rhs, ierr)
        call KSPDestroy(this%solver, ierr)

        call PetscFinalize(ierr)

    end subroutine dealloc

end module mod_linsystem
