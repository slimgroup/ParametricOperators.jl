using Tao

TEST_DISPLAY_WIDTH = 120

function adjoint_test(A::L, name::Any = nothing) where {D,R,L<:AbstractLinearOperator{D,R}}

    display_name = isnothing(name) ? id(A) : name
    s = "Running adjoint test for $(display_name) "
    nc = length(s)
    print(s)
    print(repeat(".", TEST_DISPLAY_WIDTH-nc))

    x = rand(ddt(A), Domain(A))
    y = rand(rdt(A), Range(A))
    θ = init(A)

    ỹ = A(θ...)*x
    x̃ = A(θ...)'*y
    u = real(y'*ỹ)
    v = real(x̃'*x)
    
    rat = u/v
    if rat ≈ 1
        printstyled(" PASSED.\n", color=:green)
    else
        printstyled(" FAILED.\n", color=:red)
        println("\t<y', Ax> = $(u), <A'y, x> = $(v)")
    end
end

for T in [Float16, Float32, Float64]
    adjoint_test(MatrixOperator{T}(3, 4), "MatrixOperator{$T}")
    adjoint_test(MatrixOperator{Complex{T}}(3, 4), "MatrixOperator{Complex{$T}}")
    if T != Float16
        adjoint_test(DFTOperator{Complex{T}}((3, 4, 5), (1, 2)), "DFTOperator{Complex{$T}}")
        adjoint_test(DRFTOperator{T,Complex{T}}((3, 4, 5), (1, 2)), "DRFTOperator{$T,Complex{$T}}")
    end
    adjoint_test(DRFTOperator{T,Complex{T}}((3, 4)) ⊗ MatrixOperator{Complex{T}}(3, 4), "DRFTOperator{$T,Complex{$T}}_kron_MatrixOperator{Complex{$T}}")
end