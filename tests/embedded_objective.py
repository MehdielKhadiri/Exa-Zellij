"""Exercise dtype/stride conversion in the optional Python MPI bridge."""
import numpy as np

def evaluate(x):
    return 7.0 + float(np.sum((np.asarray(x)-0.37)**2))

def evaluate_batch(rows):
    values=7.0+np.sum((np.asarray(rows)-0.37)**2,axis=1)
    padded=np.zeros(len(values)*2,dtype=np.float32)
    padded[::2]=values
    return padded[::2]  # strided float32 must safely become contiguous doubles
