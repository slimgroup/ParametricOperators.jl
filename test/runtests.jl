using LinearAlgebra
using Tao

const TEST_LINE_WIDTH = 100

function adjoint_test(A::Operator{D,R,Linear,P}, name::String) where {D<:Number,R<:Number,P<:Parametricity}

    s = "Running adjoint test for $(name) "
    print(s)
    print(repeat(".", TEST_LINE_WIDTH-length(s)))
    
    θ = init(A)

    x = rand(D, Domain(A))
    y = rand(R, Range(A))
    ỹ = A(θ)*x
    x̃ = A(θ)'*y

    u = real(ỹ'*y)
    v = real(x'*x̃)
    r = u/v

    if r ≈ 1
        printstyled(" PASSED.\n", color=:green)
    else
        printstyled(" FAILED.\n", color=:red)
        println("\t<Ax, y> = $(u), <x, Aᵀy> = $(v)")
    end
end

for T in [Float32, Float64]
    adjoint_test(RestrictionOperator{T}([17, 29, 33], [[1:5], [6:8], [12:17]]), "RestrictionOperator{$T}")
    #adjoint_test(DRFTOperator{T,Complex{T}}([16, 32, 64], [1, 2, 3]), "DRFTOperator{$T,Complex{$T}}")
    adjoint_test(MatrixOperator{T}(3, 4), "MatrixOperator{$T}")
    adjoint_test(DiagonalOperator{T}(4), "DiagonalOperator{$T}")
    adjoint_test(MatrixOperator{T}(3, 4) + MatrixOperator{T}(3, 4), "AddMatrixOperator{$T}")
    adjoint_test(MatrixOperator{T}(5, 4) * MatrixOperator{T}(4, 3), "MulMatrixOperator{$T}")
    adjoint_test(MatrixOperator{T}(5, 4) ⊗ MatrixOperator{T}(4, 3), "KronMatrixOperator{$T}")
    adjoint_test(MatrixOperator{Complex{T}}(3, 4), "MatrixOperator{Complex{$T}}")
    adjoint_test(DiagonalOperator{Complex{T}}(4), "DiagonalOperator{Complex{$T}}")
    adjoint_test(DFTOperator{Complex{T}}([16, 32, 64], [1, 3]), "DFTOperator{Complex{$T}}")
    adjoint_test(MatrixOperator{Complex{T}}(3, 4) + MatrixOperator{Complex{T}}(3, 4), "AddMatrixOperator{Complex{$T}}")
    adjoint_test(MatrixOperator{Complex{T}}(5, 4) * MatrixOperator{Complex{T}}(4, 3), "MulMatrixOperator{Complex{$T}}")
    adjoint_test(MatrixOperator{Complex{T}}(5, 4) ⊗ MatrixOperator{Complex{T}}(4, 3), "KronMatrixOperator{Complex{$T}}")
end