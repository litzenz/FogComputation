#comm_p2p_unblocking.py
# unblocking p2p: .isend() .irecv()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'a': 1, 'b': 2}
    print('process %d sends %s' %(rank, data))
    req = comm.isend(data, dest=1, tag=rank)
    #req.wait()
elif rank == 1:
    req = comm.irecv(source=0, tag=0)
    data = req.wait()
    print('process %d receives %s' %(rank, data))