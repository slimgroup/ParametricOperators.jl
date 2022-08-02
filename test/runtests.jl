using Tao, Test, LinearAlgebra

function adjoint_test(A::AbstractLinearOperator{D,R}, name::Any = nothing) where {D,R}
    op_name = isnothing(name) ? A.id : name
    print("Running adjoint test on operator $(op_name)...")
    x = rand(ddt(A), Domain(A))
    y = rand(rdt(A), Range(A))
    θs = init(A)
    ỹ = A(θs...)*x
    x̃ = A(θs...)'*y
    x2 = A(θs...)'* ỹ
    @show norm(x - x2)

    v1 = real(y'*ỹ) 
    v2 = real(x̃'*x)
    diff = abs(v1 - v2)
    rat = v1/v2
    println("\tabs(y'Ax - (A'y)'x) = $(diff)")
    println("\ty'Ax/(A'y)'x        = $(v1/v2)")
    @test rat ≈ 1
end

# Test basic operators
for T in [Float32, Float64]
    adjoint_test(MatrixOperator{T}(3, 4), "MatrixOperator{$T}")
    adjoint_test(MatrixOperator{Complex{T}}(3, 4), "MatrixOperator{Complex{$T}}")
    adjoint_test(DiagonalOperator{T}(5), "DiagonalOperator{$T}")
    adjoint_test(DFTOperator{Complex{T}}([4, 5, 6]), "DFTOperator{Complex{$T}}")
    adjoint_test(DRFTOperator{T,Complex{T}}([16, 32]), "DRFTOperator{Complex{$T}}")
    adjoint_test(RestrictionOperator{T}(10, [1:4,7:10]), "RestrictionOperator{$T}")


    # Test operator combinations
    A = MatrixOperator{T}(3, 4)
    B = MatrixOperator{T}(3, 4)
    C = MatrixOperator{T}(3, 4)
    D = A+B+C
    adjoint_test(D, "MatrixOperatorAdd{T}")

    A = MatrixOperator{T}(3, 4)
    B = MatrixOperator{T}(4, 5)
    C = MatrixOperator{T}(5, 2)
    D = A ⊗ B ⊗ C
    adjoint_test(D, "MatrixOperatorKron{T}")

    M = MatrixOperator{Complex{T}}(4, 4)
    F = DFTOperator{Complex{T}}([4])
    A = M+F
    adjoint_test(A, "MatrixDFTOperatorAdd{Complex{T}}")

    print("Running kron comparison for DFT... ")
    F1 = DFTOperator{Complex{T}}([4])
    F2 = DFTOperator{Complex{T}}([5])
    F3 = DFTOperator{Complex{T}}([6])
    Fk = F3 ⊗ F2 ⊗ F1
    Fa = DFTOperator{Complex{T}}([4, 5, 6])
    x = rand(ddt(Fa), Domain(Fa))
    yk = Fk*x
    ya = Fa*x
    @test yk ≈ ya
    println("\tnorm(yk - ya) = $(norm(yk-ya))")
end