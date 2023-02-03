using FFTW
using ParametricOperators
using Test

function adjoint_test(A::ParLinearOperator{D,R,<:ParametricOperators.Applicable,T}) where {D,R,T}
    x = rand(DDT(A), Domain(A))
    y = rand(RDT(A), Range(A))
    ỹ = A*x
    x̃ = A'*y
    u = real(x'*x̃)
    v = real(y'*ỹ)
    r = u/v
    @test r ≈ 1.0 rtol=1e-3
    
    batch_size = 10
    x = rand(DDT(A), Domain(A), batch_size)
    y = rand(RDT(A), Range(A), batch_size)
    ỹ = A*x
    x̃ = A'*y
    u = real(vec(x)'*vec(x̃))
    v = real(vec(y)'*vec(ỹ))
    r = u/v
    @test r ≈ 1.0 rtol=1e-3
end

function adjoint_test(A::ParLinearOperator{D,R,ParametricOperators.Parametric,T}) where {D,R,T}
    θ = init(A)
    At = A(θ)
    adjoint_test(At)
end

@testset "Adjoints - First Order" begin
    for T in [:Float32, :Float64]
        @eval begin
            adjoint_test(ParMatrix($T, 4, 5))
            adjoint_test(ParMatrix(Complex{$T}, 4, 5))
            #adjoint_test(ParDiagonal($T, 42))
            #adjoint_test(ParDiagonal(Complex{$T}, 42))
            #adjoint_test(ParRestriction($T, 64, [1:4, 61:64]))
            #adjoint_test(ParRestriction(Complex{$T}, 64, [1:4, 61:64]))
            adjoint_test(ParDFT(Complex{$T}, 64))
        end
    end
end

@testset "Adjoints - Second Order" begin
    # Sufficiently large matrices, else numerical errors
    A = ParMatrix(10, 20)
    B = ParMatrix(30, 50)
    C = ParMatrix(60, 90)
    K1 = C ⊗ B ⊗ A
    K2 = A ⊗ B ⊗ C
    adjoint_test(K1)
    adjoint_test(K2)

    # Map data to FFT space for multidim FFTs
    (nx, ny) = (253, 640)
    Fx = ParDFT(Float64, nx)
    Fy = ParDFT(ny)
    F = Fy ⊗ Fx
    x = rand(DDT(F), Domain(F))
    y = vec(rfft(rand(nx, ny)))
    ỹ = F*x
    x̃ = F'*y
    u = real(x'*x̃)
    v = real(y'*ỹ)
    r = u/v
    @test r ≈ 1.0 rtol=1e-3
end
