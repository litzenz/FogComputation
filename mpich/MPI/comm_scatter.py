# comm_scatter.py
#.scatter() for normal python objects

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    sdata = range(size-1)
    print('before scatter: process %d has %s' %(rank, sdata))
else:
    sdata = None
    print('before scatter: process %d has %s' %(rank, sdata))
# need object 'data' receive .bcast()
rdata = comm.scatter(sdata, root=0)
print('after bcast: process %d has %s' %(rank, rdata))