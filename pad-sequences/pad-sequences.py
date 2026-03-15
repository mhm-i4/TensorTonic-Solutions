import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len=max((len(seq) for seq in seqs),default=0)
        
    if max_len == 0:
        return np.empty((len(seqs),0))
        
    arr = np.full((len(seqs),max_len),pad_value)
    i=0
    j=0
    for seq in seqs:
        j=0
        for val in seq:
            if j >= max_len:
                break
            arr[i][j]=val
            j+=1
        i+=1
    return arr
    pass