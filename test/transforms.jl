# test that transformations behave the same as the original.

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
test_type_pairs = Vector{Tuple{Type, Type}}([
    (Float16, ComplexF16),
    (Float32, ComplexF32),
    (Float64, ComplexF64),
])

test_expressions = Vector{ParOperator}([])

for T in test_types
    i4 = ParIdentity(T, 4)
    i8 = ParIdentity(T, 8)
    m4x4 = ParMatrix(T, 4, 4)
    m4x5 = ParMatrix(T, 4, 5)
    m5x5 = ParMatrix(T, 5, 5)
    m4x8 = ParMatrix(T, 4, 8)
    m8x4 = ParMatrix(T, 8, 4)
    m8x8 = ParMatrix(T, 8, 8)
    dft4 = ParDFT(T, 4)
    dft8 = ParDFT(T, 8)
    idft4 = dft4'
    idft8 = dft8'

    push!(test_expressions, m8x4 * m4x4 * m4x8)
    push!(test_expressions, m4x8 * m8x8 * m8x4)
    push!(test_expressions, m8x4 * i4 * m4x4 * m4x8)
    push!(test_expressions, m4x8 * i8 * m8x8 * m8x4)
    push!(test_expressions, i8 * (m8x8 * m8x8) ⊗ (m8x4 * m4x8))
    push!(test_expressions, i8 * (m8x8 * m8x8) ⊗ (m4x8 * m8x4))
    push!(test_expressions, idft4 * dft4 * m4x4)
end

for (RT, CT) in test_type_pairs
    push!(test_expressions, ParMatrix(CT, 4, 4) * ParMatrix(CT, 4, 5) * ParDFT(RT, 8) * ParMatrix(RT, 8, 4) * ParDiagonal(RT, 4))
end


@testset "Transforms - first level (immediate neighbors)" begin
    for T in test_expressions
        @eval begin
            T0=$T
            # "good" data
            θ = init(T0)
            x = rand(DDT(T0), Domain(T0))
            # println("T0 is ", T0)
            # println("Domain(T0) is ", Domain(T0))
            # println("Range(T0) is ", Range(T0))
            # println("typeof(θ) is ", typeof(θ))
            # println("typeof(x) is ", typeof(x))
            # println("DDT(T0) is ", DDT(T0))
            # println("size(x) is ", size(x))
            # println("θ is ", θ)
            # println("length(θ) is ", length(θ))
            y = T0(θ)*x
            # println("typeof(y) is ", typeof(y))
            # println("RDT(T0) is ", RDT(T0))
            # println("size(y) is ", size(y))
            for T1 in transforms(T0)
                y1 = T1(θ)*x
                @test y ≈ y1
            end
            # println()
        end
    end
end

@testset "Transforms - second level (neighbors of neighbors)" begin
    for T in test_expressions
        @eval begin
            T0=$T
            # "good" data
            θ = init(T0)
            x = rand(DDT(T0), Domain(T0))
            y = T0(θ)*x

            for T1 in transforms(T0)
                for T2 in transforms(T1)
                    y2 = T2(θ)*x
                    @test y ≈ y2
                end
            end
        end
    end
end

@testset "Transforms - third level" begin
    for T in test_expressions
        @eval begin
            T0=$T
            # "good" data
            θ = init(T0)
            x = rand(DDT(T0), Domain(T0))
            y = T0(θ)*x

            for T1 in transforms(T0)
                for T2 in transforms(T1)
                    for T3 in transforms(T2)
                        y3 = T3(θ)*x
                        @test y ≈ y3
                    end
                end
            end
        end
    end
end
