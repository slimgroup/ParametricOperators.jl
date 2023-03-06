using JSON
using ParametricOperators
using Test

@testset "JSON" begin
    test_types = Vector{Type}([
        Float16,
        Float32,
        Float64,
        ComplexF16,
        ComplexF32,
        ComplexF64,
    ])
    test_ops = vcat(
        # base ops
        collect(ParDiagonal(T, 4) for T in test_types),
        collect(ParDFT(T, 4) for T in test_types),
        collect(ParIdentity(T, 4) for T in test_types),
        collect(ParMatrix(T, 3, 4) for T in test_types), # random uuids
        collect(ParMatrix(T, 3, 4, string(T)) for T in test_types), # string ids
        collect(ParRestriction(T, 4, [1:4, 5:8]) for T in test_types),

        # composite ops
        collect(ParAdjoint(ParIdentity(T, 4)) for T in test_types),
        collect(ParCompose(ParDiagonal(T, 4), ParIdentity(T, 4)) for T in test_types),
        collect(ParKron(ParDiagonal(T, 4), ParIdentity(T, 4)) for T in test_types),
        )

    for op in test_ops
        json = to_json(op)
        # it's a string
        @test typeof(json) == String
        # it's a json string containing a dict
        @test typeof(JSON.parse(json)) == Dict{String, Any}

        decoded = from_json(json)
        # type equality
        @test typeof(op) == typeof(decoded)
        # round trip equality
        @test string(op) == string(decoded)
        # double round trip equality
        @test string(op) == string(from_json(to_json(from_json(to_json(op)))))
    end
end
