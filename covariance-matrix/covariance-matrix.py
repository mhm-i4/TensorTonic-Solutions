import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X=np.array(X)
    if X.ndim == 1:
        return None
    N,d=X.shape
    
    if N<2:
        return None
        
    u=np.mean(X,axis=0)
    
    x=X-u
    E=(x.T @ x)/(N-1)
    return E
    # Write code here
    pass