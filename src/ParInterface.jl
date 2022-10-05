export joWrapper

struct joWrapper{D,R,F<:joAbstractLinearOperator{D,R}} <: ParOperator{D,R,Linear,NonParametric,External}
    op::F
    id::ID
    joWrapper(A::joAbstractLinearOperator) = new{deltype(A),reltype(A),typeof(A)}(A, uuid4(GLOBAL_RNG))
end

Domain(A::joWrapper) = A.op.n
Range(A::joWrapper) = A.op.m
id(A::joWrapper) = A.id

adjoint(A::joWrapper) = joWrapper(adjoint(A.op))

(A::joWrapper{D,R,F})(x::X) where {D,R,F,X<:AbstractVector{D}} = A.op*x