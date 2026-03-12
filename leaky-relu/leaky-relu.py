import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x=np.asarray(x)
    new_arr=np.where(x<0,x*alpha,x)
    return new_arr
    pass