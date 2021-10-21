program main_axi
use mod_linsystem
implicit none

type(linsystem) :: test
integer :: i, j
double precision, dimension(:), allocatable :: rhs, sol
integer :: myrank, nprocs, ierr, global_index
integer, dimension(:, :), allocatable :: indices
double precision, dimension(:, :), allocatable :: values
double precision, parameter :: h = 0.01
integer :: nnx, nny, ncx, ncy
double precision :: xmin, xmax, Lx, dx, ymin, ymax, Ly, dy, scale
double precision :: x0, y0, x01, y01, sigma_x, sigma_y, ampl
integer :: ix, iy
double precision, dimension(:, :), allocatable :: x, y
double precision, dimension(:), allocatable :: y_flatten
integer :: ierror

! Some strings
CHARACTER(LEN=len_str) :: arg, str, casename                          !< Argument of command line

! For performance monitoring
double precision :: t1, t2, t_tot, t_avg, t_best, t_var, t_stddev
double precision, dimension(:), allocatable :: times
integer :: it, solver_its

! Resolution as command line arguments and name of solver
CALL get_command_argument(1, arg, len_arg, ierror)
read(arg, '(I10)') nnx
CALL get_command_argument(2, arg, len_arg, ierror)
read(arg, '(I10)') nny
CALL get_command_argument(3, casename, len_arg, ierror)

call MPI_INIT(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

! Mesh properties
ncx = nnx - 1
xmin = 0.0d0
xmax = 4.0d-3
Lx = xmax - xmin
dx = Lx / ncx

ncy = nny - 1
ymin = 0.0d0
ymax = 1.0d-3
Ly = ymax - ymin
dy = Ly / ncy

scale = dx * dy

! Create geometry vectors
allocate(x(nnx, nny), y(nnx, nny), y_flatten(nnx * nny))
do i = 1, nnx
    do j = 1, nny
        x(i, j) = (i - 1) * dx
        y(i, j) = (j - 1) * dy
        y_flatten((j - 1) * nnx + i) = (j - 1) * dy
    end do
end do

! Initiailize matrix and print
call test%initSystem(nnx * nny, casename)

! Non null vectors
allocate(indices(test%local_size, 5), values(test%local_size, 5))

do i = 1, test%local_size
    global_index = i + test%istart - 1
    if (isAxis(global_index)) then
        values(i, 1) = - (2 / dx**2 + 4 / dy**2) * scale
        values(i, 2) = 1 / dx**2 * scale
        values(i, 3) = 1 / dx**2 * scale
        values(i, 4) = 4 / dy**2 * scale
        indices(i, 1) = global_index
        indices(i, 2) = global_index + 1
        indices(i, 3) = global_index - 1
        indices(i, 4) = global_index + nnx
        indices(i, 5) = -1
    else if (isBoundary(global_index)) then
        values(i, 1) = 1
        indices(i, 1) = global_index
        indices(i, 2:) = -1
    else
        values(i, 1) = - (2 / dx**2 + 2 / dy**2) * scale
        values(i, 2) = 1 / dx**2 * scale
        values(i, 3) = 1 / dx**2 * scale
        values(i, 4) = (1 + dy / (2 * y_flatten(i))) / dy**2 * scale
        values(i, 5) = (1 - dy / (2 * y_flatten(i))) / dy**2 * scale

        indices(i, 1) = global_index
        ! Classical matrix
        indices(i, 2) = global_index + 1
        indices(i, 3) = global_index - 1
        indices(i, 4) = global_index + nnx
        indices(i, 5) = global_index - nnx
    end if
end do

call test%assembleMatrix(indices, values)

! Initiailize solver and print
call test%initSolver()
call test%solverInfo()

! Initialize rhs and sol with zero dirichlet boundary condition
allocate(rhs(test%local_size), sol(test%local_size))

! Gaussian RHS in axisym
sigma_x = 1d-3
sigma_y = 1d-3
x0 = 2.0d-3
y0 = 0.0d0
ampl = 1d16 * 1.602176634e-19 / 8.8541878128e-12
do i = 1, test%local_size
    global_index = i + test%istart - 1
    if (global_index >= (nny - 1) * nnx + 1 .or. &
            mod(global_index, nnx) == 0 .or. mod(global_index, nnx) == 1) then
        rhs(i) = 0.0d0
    else
        ix = mod(global_index, nnx) + 1
        iy = global_index / nnx + 1
        rhs(i) = - ampl * exp(-(x(ix, iy) - x0)**2 / sigma_x**2 - (y(ix, iy) - y0)**2 / sigma_y**2) * scale
    end if
end do

! Solve the linear system
solver_its = 20
allocate(times(solver_its))
do it = 1, solver_its
    call PetscTime(t1, ierr)
    call test%solve(rhs, sol)
    call PetscTime(t2, ierr)
    times(it) = t2 - t1
end do
t_best = MINVAL(times)
t_tot = SUM(times)
t_avg = t_tot / solver_its
t_var = 0.0
do it = 1, solver_its
   t_var = t_var + (times(it) - t_avg)**2
end do
t_var = t_var / (solver_its - 1)
t_stddev = SQRT(t_var)

! See the content of the vectors
call test%solveInfo()

! Print solver info
if (test%myrank == 0) then
    open(11, file=test%casename, status="old", position="append", action="write")
    write(11, *) '*-------------------------------'
    write(11, "('nnodes       = ', I0)") nnx * nny
    write(11, "('elapsed_time = ', ES10.3)") t_tot
    write(11, "('best_time    = ', ES10.3)") t_best
    write(11, "('average_time = ', ES10.3)") t_avg
    write(11, "('stddev_time  = ', ES10.3)") t_stddev
    close(11)
end if

! ! See the matrix and vectors
! call test%vectorsInfo()
! call test%matrixInfo()

call test%dealloc()

contains

! Indicates if global_index is a symmetry axis node or not
function isAxis(global_index)
    implicit none
    integer, intent(in) :: global_index
    logical :: isAxis

    isAxis = global_index < nnx .and. global_index > 0

end function

! Indicates if global_index is a boundary or not
function isBoundary(global_index)
    implicit none
    integer, intent(in) :: global_index
    logical :: isBoundary

    isBoundary = global_index >= (nny - 1) * nnx + 1 .or. &
            mod(global_index, nnx) == 0 .or. mod(global_index, nnx) == 1

end function

end program main_axi