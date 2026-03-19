import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a=np.array(A)
    m=len(A[0])
    n=len(A)
    arr=np.zeros((m,n))
    
    for i in range(n):
        for j in range(m):
            arr[j][i]=a[i][j]

    return arr
    # Write code here
    pass
