export ParAdjoint

struct ParAdjoint{D,R,P,O,F} <: ParLinearOperator{R,D,P,HigherOrder}
    op::F
    ParAdjoint(op) = new{DDT(op),RDT(op),parametricity(op),order(op),typeof(op)}(op)
end

Domain(A::ParAdjoint) = Range(A.op)
Range(A::ParAdjoint) = Domain(A.op)
children(A::ParAdjoint) = [A.op]
id(A::ParAdjoint) = "adjoint_$(id(A.op))"

adjoint(A::ParLinearOperator) = ParAdjoint(A)
adjoint(A::ParAdjoint) = A.op
(A::ParAdjoint{D,R,Parametric,O,F})(θ) where {D,R,O,F} = ParAdjoint(A(θ))