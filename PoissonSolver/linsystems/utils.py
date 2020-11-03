def print_mat(A, matname):
    print(f'Matrice {matname}')
    n = len(A[:, 0])
    for i in range(n):
        row = ''
        for j in range(n):
            row += f'{A[i, j]:>6.2f} '
        print(row)