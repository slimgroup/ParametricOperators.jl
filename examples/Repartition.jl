# Run using mpiexecjl -n 6 julia examples/Repartition.jl

using MPI
using Tao

MPI.Init()

world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
world_size = MPI.Comm_size(MPI.COMM_WORLD)
@assert world_size == 6

comm_x = MPI.Cart_create(MPI.COMM_WORLD, [2, 3], [0, 0], 0)
comm_y = MPI.Cart_create(MPI.COMM_WORLD, [3, 2], [0, 0], 0)
shape  = [8, 8]

R = Tao.RepartitionOperator{Float32}(comm_x, comm_y, shape)
rank = MPI.Comm_rank(R.comm)
x = ddt(R).(fill(rank+1, Domain(R)))

b = IOBuffer()
show(b, "text/plain", reshape(x, R.local_shape_x...))
print_seq("$(rank) -> $(String(take!(b)))", R.comm)

y = R*x

MPI.Barrier(R.comm)

b = IOBuffer()
show(b, "text/plain", reshape(y, R.local_shape_y...))
print_seq("$(rank) -> $(String(take!(b)))", R.comm)

MPI.Finalize()