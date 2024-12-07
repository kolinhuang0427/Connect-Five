from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank (0 or 1 in our case)
size = comm.Get_size()  # Total number of processes

print(f"Hello from rank {rank}, total number of processes: {size}")