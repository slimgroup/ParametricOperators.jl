using MPI
using ParametricOperators
using Test
using LinearAlgebra

MPI.Init()

world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
world_size = MPI.Comm_size(MPI.COMM_WORLD)
@assert world_size == 2 "Program must be run with 2 workers"

T = Float64

network = ParIdentity(T, 2) ⊗ ParIdentity(T, 2) ⊗ ParIdentity(T, 2)
network = distribute(network, [1, 2, 1])

x = reshape(float(1:8), 2, 2, 2)
x = vec(x[:,world_rank+1,:])

y_out = network * x
@test norm(x - y_out) == 0

MPI.Finalize()
