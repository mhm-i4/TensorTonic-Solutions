import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    y=x_t@Wx + h_prev@Wh + b
    h_t=np.tanh(y)
    return h_t
    # Write code here
    pass
