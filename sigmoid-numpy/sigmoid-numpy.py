import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    #convert the list into numpy array
    inpt=np.array(x)
    return 1.0/(1+np.exp(-inpt))
    pass