# test AST hashing / uniqueness

using ParametricOperators
using Test

# basics
i4    = ParIdentity(ComplexF32, 4)
i4_2  = ParIdentity(ComplexF32, 4)
i4c64 = ParIdentity(ComplexF64, 4)
i8    = ParIdentity(ComplexF32, 8)
i8_2  = ParIdentity(ComplexF32, 8)
i8c64 = ParIdentity(ComplexF64, 8)
d4    = ParDFT(ComplexF32, 4)
d4_2  = ParDFT(ComplexF32, 4)
d4c64 = ParDFT(ComplexF64, 4)
d8    = ParDFT(ComplexF32, 8)
d8_2  = ParDFT(ComplexF32, 8)
d8c64 = ParDFT(ComplexF64, 8)
m4x4    = ParMatrix(ComplexF32, 4, 4)
m4x4_2  = ParMatrix(ComplexF32, 4, 4)
m4x4c64 = ParMatrix(ComplexF64, 4, 4)
m4x8    = ParMatrix(ComplexF32, 4, 8)
m4x8_2  = ParMatrix(ComplexF32, 4, 8)
m4x8c64 = ParMatrix(ComplexF64, 4, 8)
m8x4    = ParMatrix(ComplexF32, 8, 4)
m8x4_2  = ParMatrix(ComplexF32, 8, 4)
m8x4c64 = ParMatrix(ComplexF64, 8, 4)
m8x8    = ParMatrix(ComplexF32, 8, 8)
m8x8_2  = ParMatrix(ComplexF32, 8, 8)
m8x8c64 = ParMatrix(ComplexF64, 8, 8)

# composites
c44   = m4x4 * m4x4_2
c44_2 = m4x4 * m4x4_2
c88   = m8x8 * m8x8_2
c88_2 = m8x8 * m8x8_2
c484  = m4x8 * m8x4
c848  = m8x4 * m4x8
c4884  = m4x8 * m8x8 * m8x4
c8448  = m8x4 * m4x4 * m4x8
c48i4  = m4x8 * i8 * m8x4
c84i8  = m8x4 * i4 * m4x8
k44     = m4x4 ⊗ m4x4_2
k44_2   = m4x4 ⊗ m4x4_2
k48     = m4x4 ⊗ m4x8
k48_2   = m4x4 ⊗ m4x8
k84     = m8x8 ⊗ m4x4
k84_2   = m8x8 ⊗ m4x4
k88     = m8x8 ⊗ m8x8_2
k88_2   = m8x8 ⊗ m8x8_2
ki444   = i4 ⊗ m4x4 ⊗ m4x4_2
ki444_2 = i4 ⊗ m4x4 ⊗ m4x4_2
ki848   = i8 ⊗ m4x4 ⊗ m4x8
ki848_2 = i8 ⊗ m4x4 ⊗ m4x8

# direct comparisons between basic types
@testset "basics are equal to themselves" begin
    @test merkle_hash(i4) == merkle_hash(i4)
    @test merkle_hash(i8) == merkle_hash(i8)
    @test merkle_hash(d4) == merkle_hash(d4)
    @test merkle_hash(d8) == merkle_hash(d8)
    @test merkle_hash(m4x4) == merkle_hash(m4x4)
    @test merkle_hash(m4x8) == merkle_hash(m4x8)
    @test merkle_hash(m8x4) == merkle_hash(m8x4)
    @test merkle_hash(m8x8) == merkle_hash(m8x8)
end

@testset "basics don't match each other" begin
    @test merkle_hash(i4) != merkle_hash(i8)
    @test merkle_hash(i4) != merkle_hash(d4)
    @test merkle_hash(i4) != merkle_hash(d8)
    @test merkle_hash(i4) != merkle_hash(m4x4)
    @test merkle_hash(i4) != merkle_hash(m4x8)
    @test merkle_hash(i4) != merkle_hash(m8x4)
    @test merkle_hash(i4) != merkle_hash(m8x8)

    @test merkle_hash(i8) != merkle_hash(d4)
    @test merkle_hash(i8) != merkle_hash(d8)
    @test merkle_hash(i8) != merkle_hash(m4x4)
    @test merkle_hash(i8) != merkle_hash(m4x8)
    @test merkle_hash(i8) != merkle_hash(m8x4)
    @test merkle_hash(i8) != merkle_hash(m8x8)

    @test merkle_hash(d4) != merkle_hash(d8)
    @test merkle_hash(d4) != merkle_hash(m4x4)
    @test merkle_hash(d4) != merkle_hash(m4x8)
    @test merkle_hash(d4) != merkle_hash(m8x4)
    @test merkle_hash(d4) != merkle_hash(m8x8)

    @test merkle_hash(d8) != merkle_hash(m4x4)
    @test merkle_hash(d8) != merkle_hash(m4x8)
    @test merkle_hash(d8) != merkle_hash(m8x4)
    @test merkle_hash(d8) != merkle_hash(m8x8)

    @test merkle_hash(m4x4) != merkle_hash(m4x8)
    @test merkle_hash(m4x4) != merkle_hash(m8x4)
    @test merkle_hash(m4x4) != merkle_hash(m8x8)

    @test merkle_hash(m4x8) != merkle_hash(m8x4)
    @test merkle_hash(m4x8) != merkle_hash(m8x8)

    @test merkle_hash(m8x4) != merkle_hash(m8x8)
end

@testset "identities with the same params are equal" begin
    @test merkle_hash(i4) == merkle_hash(i4_2)
    @test merkle_hash(i8) == merkle_hash(i8_2)
end

@testset "identities with different types are different" begin
    @test merkle_hash(i4) != merkle_hash(i4c64)
    @test merkle_hash(i8) != merkle_hash(i8c64)
end

@testset "DFTs with the same params are equal" begin
    @test merkle_hash(d4) == merkle_hash(d4_2)
    @test merkle_hash(d8) == merkle_hash(d8_2)
end

@testset "DFTs with different types are different" begin
    @test merkle_hash(d4) != merkle_hash(d4c64)
    @test merkle_hash(d8) != merkle_hash(d8c64)
end

@testset "matrices with the same params are equivalent" begin
    # they may have different params, but they have the same cost.
    @test merkle_hash(m4x4) == merkle_hash(m4x4_2)
    @test merkle_hash(m4x8) == merkle_hash(m4x8_2)
    @test merkle_hash(m8x4) == merkle_hash(m8x4_2)
    @test merkle_hash(m8x8) == merkle_hash(m8x8_2)
end

@testset "matrices with different types are different" begin
    # they may have different params, but they have the same cost.
    @test merkle_hash(m4x4) != merkle_hash(m4x4c64)
    @test merkle_hash(m4x8) != merkle_hash(m4x8c64)
    @test merkle_hash(m8x4) != merkle_hash(m8x4c64)
    @test merkle_hash(m8x8) != merkle_hash(m8x8c64)
end

@testset "compositions of the same elements are equivalent" begin
    @test merkle_hash(c44) == merkle_hash(c44_2)
    @test merkle_hash(c88) == merkle_hash(c88_2)
end

@testset "compositions of different elements are different" begin
    @test merkle_hash(c44) != merkle_hash(c88)
    @test merkle_hash(c44) != merkle_hash(c484)
    @test merkle_hash(c44) != merkle_hash(c848)
    @test merkle_hash(c44) != merkle_hash(c4884)
    @test merkle_hash(c44) != merkle_hash(c8448)
    @test merkle_hash(c44) != merkle_hash(c48i4)
    @test merkle_hash(c44) != merkle_hash(c84i8)

    @test merkle_hash(c88) != merkle_hash(c484)
    @test merkle_hash(c88) != merkle_hash(c848)
    @test merkle_hash(c88) != merkle_hash(c4884)
    @test merkle_hash(c88) != merkle_hash(c8448)
    @test merkle_hash(c88) != merkle_hash(c48i4)
    @test merkle_hash(c88) != merkle_hash(c84i8)

    @test merkle_hash(c484) != merkle_hash(c848)
    @test merkle_hash(c484) != merkle_hash(c4884)
    @test merkle_hash(c484) != merkle_hash(c8448)
    @test merkle_hash(c484) != merkle_hash(c48i4)
    @test merkle_hash(c484) != merkle_hash(c84i8)

    @test merkle_hash(c848) != merkle_hash(c4884)
    @test merkle_hash(c848) != merkle_hash(c8448)
    @test merkle_hash(c848) != merkle_hash(c48i4)
    @test merkle_hash(c848) != merkle_hash(c84i8)

    @test merkle_hash(c4884) != merkle_hash(c8448)
    @test merkle_hash(c4884) != merkle_hash(c48i4)
    @test merkle_hash(c4884) != merkle_hash(c84i8)

    @test merkle_hash(c8448) != merkle_hash(c48i4)
    @test merkle_hash(c8448) != merkle_hash(c84i8)

    @test merkle_hash(c48i4) != merkle_hash(c84i8)
end

@testset "krons of the same elements are equivalent" begin
    @test merkle_hash(k44)   == merkle_hash(k44_2)
    @test merkle_hash(k48)   == merkle_hash(k48_2)
    @test merkle_hash(k84)   == merkle_hash(k84_2)
    @test merkle_hash(k88)   == merkle_hash(k88_2)
    @test merkle_hash(ki444) == merkle_hash(ki444_2)
    @test merkle_hash(ki848) == merkle_hash(ki848_2)
end

@testset "krons of different elements are different" begin
    @test merkle_hash(k44) != merkle_hash(k48)
    @test merkle_hash(k44) != merkle_hash(k84)
    @test merkle_hash(k44) != merkle_hash(k88)
    @test merkle_hash(k44) != merkle_hash(ki444)
    @test merkle_hash(k44) != merkle_hash(ki848)

    @test merkle_hash(k48) != merkle_hash(k84)
    @test merkle_hash(k48) != merkle_hash(k88)
    @test merkle_hash(k48) != merkle_hash(ki444)
    @test merkle_hash(k48) != merkle_hash(ki848)

    @test merkle_hash(k84) != merkle_hash(k88)
    @test merkle_hash(k84) != merkle_hash(ki444)
    @test merkle_hash(k84) != merkle_hash(ki848)

    @test merkle_hash(k88) != merkle_hash(ki444)
    @test merkle_hash(k88) != merkle_hash(ki848)

    @test merkle_hash(ki444) != merkle_hash(ki848)
end
