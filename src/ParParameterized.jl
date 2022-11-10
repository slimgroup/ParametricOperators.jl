export ParParameterized

struct ParParameterized{D,R,L,F,V} <: ParOperator{D,R,L,Parameterized,HigherOrder}
    op::F
    θ::V
    ParParameterized(op, θ) = new{DDT(op),RDT(op),linearity(op),typeof(op),typeof(θ)}(op, θ)
end

Domain(A::ParParameterized) = Domain(A.op)
Range(A::ParParameterized) = Range(A.op)
children(A::ParParameterized) = [A.op]
children_symbol(::ParParameterized) = :op
id(A::ParParameterized) = "parameterized_$(id(A.op))"

(A::ParOperator{D,R,L,Parametric,FirstOrder})(θ) where {D,R,L} = ParParameterized(A, θ)