export ParParameterized

struct ParParameterized{D,R,L,F,V} <: ParOperator{D,R,L,Parameterized,Internal}
    op::F
    θ::V
    ParParameterized(A::ParOperator, θ) =
        new{DDT(A),RDT(A),linearity(A),typeof(A),typeof(θ)}(A, θ)
end

Domain(A::ParParameterized) = Domain(A.op)
Range(A::ParParameterized) = Range(A.op)
children(A::ParParameterized) = [A.op]
params(A::ParParameterized) = A.θ
id(A::ParParameterized) = "adjoint_$(id(A.op))"

(A::ParOperator{D,R,L,Parametric,T})(θ) where {D,R,L,T} = ParParameterized(A, θ)
(A::ParParameterized{Nothing,R,L,F,V})(x) where {R,L,F,V} = A.op(x, A.θ)

adjoint(A::ParParameterized) = ParParameterized(adjoint(A.op), A.θ)