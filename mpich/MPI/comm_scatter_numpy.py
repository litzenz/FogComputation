from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# on each process
sdata = None
if rank == 0:
    sdata = np.random.normal(0, 1, size=(size, 3))
    
# on each process:
print('before scatter: process %d has\n%s' %(rank, sdata))
rdata = np.zeros(3)
comm.Scatter(sdata, rdata, root=0)
print('after bcast: process %d has %s' %(rank, rdata))