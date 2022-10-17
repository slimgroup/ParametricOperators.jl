using ParametricOperators, Test, Printf

const LINE_WIDTH = 120

function adjoint_test(A::ParOperator{D,R,Linear,Parametric,T}; stol=1f-3) where {D,R,T}

    x = rand(DDT(A), Domain(A))
    y = rand(RDT(A), Range(A))
    θ = init(A)

    ỹ = A(θ)*x
    x̃ = A'(θ)*y
    u = real(x̃'*x)
    v = real(ỹ'*y)
    r = u/v

    @printf("%11s, %11s, %11s\n", "<Aᵀx,y>", "<Ax,y>", "r")
    @printf("%5.5e, %5.5e, %1.9f\n", u, v, r)
    @test isapprox(r, 1.0, atol=stol)
end

function adjoint_test(A::ParOperator{D,R,Linear,<:Applicable,T}; stol=1f-2) where {D,R,T}

    x = rand(DDT(A), Domain(A))
    y = rand(RDT(A), Range(A))

    ỹ = A*x
    x̃ = A'*y
    u = real(x̃'*x)
    v = real(ỹ'*y)
    r = u/v

    @printf("%11s, %11s, %11s\n", "<Aᵀx,y>", "<Ax,y>", "r")
    @printf("%5.5e, %5.5e, %1.9f\n", u, v, r)
    @test isapprox(r, 1.0, atol=stol)
end