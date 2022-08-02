# Run using mpiexecjl -n 12 julia test/runtest_mpi.jl

using Tao, MPI, Test, LinearAlgebra

MPI.Init()
world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
world_size = MPI.Comm_size(MPI.COMM_WORLD)

@assert world_size >= 12

function adjoint_test_mpi(A::AbstractLinearOperator{D,R}, comm::MPI.Comm, name::Any = nothing) where {D,R}
    op_name = isnothing(name) ? A.id : name
    rank = MPI.Comm_rank(comm)
    println0("Running distributed adjoint test on operator $(op_name)...", comm)
    x = rand(ddt(A), Domain(A))
    y = rand(rdt(A), Range(A))
    θs = init(A)
    ỹ = A(θs...)*x
    x̃ = A(θs...)'*y
    x2 = A(θs...)'* ỹ

    v1 = [real(y'*ỹ)]
    v2 = [real(x̃'*x)]
    xdiff = [sum((x2-x).^2)]
    MPI.Reduce!(v1, MPI.MPI_SUM, 0, comm)
    MPI.Reduce!(v2, MPI.MPI_SUM, 0, comm)
    MPI.Reduce!(xdiff, MPI.MPI_SUM, 0, comm)
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        v1 = v1[1]
        v2 = v2[1]
        diff = abs(v1 - v2)
        rat = v1/v2
        nm = sqrt(xdiff[1])
        println("$(rank) -> $(nm)\n\tabs(y'Ax - (A'y)'x) = $(diff)\n\ty'Ax/(A'y)'x        = $(v1/v2)");
        @test rat ≈ 1
    end
end

for T in [Float32, Float64]
    for (shape, dims_x, dims_y) in [([8, 10], [2, 3], [3, 2]), ([16, 23, 37], [2, 2, 3], [3, 2, 2])]
        np = prod(dims_x)
        color = world_rank < np
        comm = MPI.Comm_split(MPI.COMM_WORLD, color, world_rank)
        comm_x = MPI.Cart_create(comm, dims_x, repeat([0], length(dims_x)), 0)
        comm_y = MPI.Cart_create(comm, dims_y, repeat([0], length(dims_y)), 0)
        R = Tao.RepartitionOperator{T}(comm_x, comm_y, shape)
        adjoint_test_mpi(R, comm, "RestrictionOperator{$T}_$(shape)_$(dims_x)_$(dims_y)")
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

MPI.Finalize()