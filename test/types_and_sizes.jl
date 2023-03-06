# test that DDT(), RDT(), Domain() and Range() are correct for basic ops

using ParametricOperators
using Test

test_types = Vector{Type}([
    # Float16,
    Float32,
    Float64,
    # ComplexF16,
    ComplexF32,
    ComplexF64,
])
test_ops = Vector{ParOperator}([])

for T in test_types
    push!(test_ops, ParIdentity(T, 4))
    push!(test_ops, adjoint(ParIdentity(T, 4)))
    push!(test_ops, ParDiagonal(T, 4))
    push!(test_ops, adjoint(ParDiagonal(T, 4)))
    push!(test_ops, ParMatrix(T, 4, 4))
    push!(test_ops, adjoint(ParMatrix(T, 4, 4)))
    push!(test_ops, ParMatrix(T, 4, 8))
    push!(test_ops, adjoint(ParMatrix(T, 4, 8)))
    push!(test_ops, ParDFT(T, 4))
    push!(test_ops, adjoint(ParDFT(T, 4)))
    push!(test_ops, ParDFT(T, 8))
    push!(test_ops, adjoint(ParDFT(T, 8)))
    push!(test_ops, ParRestriction(T, 8, [1:2]))
    push!(test_ops, ParRestriction(T, 8, [1:2, 7:8]))
    push!(test_ops, adjoint(ParRestriction(T, 8, [1:2])))
    push!(test_ops, adjoint(ParRestriction(T, 8, [1:2, 7:8])))
end

@testset "$op" for op in test_ops
    @eval begin
        op=$op
        # "good" data
        θ = init(op)
        xv = rand(DDT(op), Domain(op))      # vector input
        xm = rand(DDT(op), (Domain(op), 2)) # matrix input
        if length(θ) > 0
            yv = op(θ)*xv
            ym = op(θ)*xm
        else
            # operator is not parameterized
            yv = op*xv
            ym = op*xm
        end
        @test size(xv) == (Domain(op),)
        @test size(yv) == (Range(op),)
        @test size(xm) == (Domain(op), 2)
        @test size(ym) == (Range(op), 2)
        @test typeof(xv) == Vector{DDT(op)}
        @test typeof(yv) == Vector{RDT(op)}
    end
end
