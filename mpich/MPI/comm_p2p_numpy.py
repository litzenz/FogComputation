# .Send(), .Recv(), .Bcast(), .Scatter, .Gather(): derectly communicate
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np

if rank==0:
    data = np.arange(10, dtype='i')
    comm.Send(data, dest=1, tag=1)
    print('process %d sends %s' %(rank, data))
elif rank==1:
    data = np.empty(10, dtype='i')
    comm.Recv(data, source=0, tag=1)
    print('process %d receive %s' %(rank, data))