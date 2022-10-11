export ParAdjoint

struct ParAdjoint{D,R,P,F} <: ParOperator{R,D,Linear,P,Internal}
    op::F
    ParAdjoint(A::ParOperator{D,R,Linear,P,T}) where {D,R,P,T} = new{D,R,P,typeof(A)}(A)
end

Domain(A::ParAdjoint) = Range(A.op)
Range(A::ParAdjoint) = Domain(A.op)
children(A::ParAdjoint) = [A.op]
id(A::ParAdjoint) = "adjoint_$(id(A.op))"

adjoint(A::ParOperator{D,R,Linear,P,T}) where {D,R,P,T} = ParAdjoint(A)

(A::ParAdjoint{D,R,P,F})(x::X) where {D,R,P<:Applicable,F,X<:AbstractVecOrMat{R}} = A.op(x)