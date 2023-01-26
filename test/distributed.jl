using MPI
using ParametricOperators
using Test

function adjoint_test_distributed(
        A::ParLinearOperator{D,R,<:ParametricOperators.Applicable,T},
        comm_in::MPI.Comm,
        comm_out::MPI.Comm
) where {D,R,T}

    if comm_in == MPI.COMM_NULL || comm_out == MPI.COMM_NULL
        return
    end

    x = rand(DDT(A), Domain(A))
    y = rand(RDT(A), Range(A))
    ỹ = A*x
    x̃ = A'*y
    u = MPI.Reduce(real(x'*x̃), MPI.SUM, comm_in)
    v = MPI.Reduce(real(y'*ỹ), MPI.SUM, comm_out)
    u = MPI.bcast(u, MPI.COMM_WORLD)
    v = MPI.bcast(v, MPI.COMM_WORLD)
    r = u/v
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test r ≈ 1.0 rtol=1e-3
    end
end

function adjoint_test_distributed(
    A::ParLinearOperator{D,R,ParametricOperators.Parametric,T},
    comm_in::MPI.Comm,
    comm_out::MPI.Comm
) where {D,R,T}
    θ = init(A)
    At = A(θ)
    adjoint_test_distributed(At, comm_in, comm_out)
end

MPI.Init()

world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
world_size = MPI.Comm_size(MPI.COMM_WORLD)
@assert world_size == 8 "Program must be run with 8 workers"

if world_rank == 0
    @testset "Repartitions" begin
        comm_in = MPI.Cart_create(MPI.COMM_WORLD, Int32.([2, 2, 2]))
        comm_out = MPI.Cart_create(MPI.COMM_WORLD, Int32.([8, 1, 1]))
        R = ParRepartition(Float32, comm_in, comm_out, (10, 10, 10))
        adjoint_test_distributed(R, comm_in, comm_out)
    end
else
    comm_in = MPI.Cart_create(MPI.COMM_WORLD, Int32.([2, 2, 2]))
    comm_out = MPI.Cart_create(MPI.COMM_WORLD, Int32.([8, 1, 1]))
    R = ParRepartition(Float32, comm_in, comm_out, (10, 10, 10))
    adjoint_test_distributed(R, comm_in, comm_out)
end

MPI.Finalize()
