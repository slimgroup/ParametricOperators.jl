export ParAdjoint

struct ParAdjoint{D,R,P,F} <: ParLinearOperator{R,D,P,HigherOrder}
    op::F
    ParAdjoint(op) = new{DDT(op),RDT(op),parametricity(op),typeof(op)}(op)
end

Domain(A::ParAdjoint) = Range(A.op)
Range(A::ParAdjoint) = Domain(A.op)
children(A::ParAdjoint) = [A.op]
id(A::ParAdjoint) = "adjoint_$(id(A.op))"

adjoint(A::ParLinearOperator) = ParAdjoint(A)
adjoint(A::ParAdjoint) = A.op
(A::ParAdjoint{D,R,Parametric,F})(θ) where {D,R,F} = ParAdjoint(A.op(θ))