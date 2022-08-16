import Base.*
import Base.adjoint

abstract type AbstractLinearOperator{D,R} <: AbstractOperator{D,R} end

*(A::AbstractLinearOperator{D,R}, ::AbstractVector{D}) where {D,R} = throw(TaoException("mvp() is not implemented for $(typeof(A))"))
(A::AbstractLinearOperator{D,R})(x::AbstractVector{D}, pv::ParameterVector) where {D,R} = A(pv)*x

struct LinearOperatorAdjoint{D,R,L<:AbstractLinearOperator{D,R}} <: AbstractLinearOperator{R,D}
    inner::L
end

adjoint(A::AbstractLinearOperator{D,R}) where {D,R} = LinearOperatorAdjoint{D,R,typeof(A)}(A)

Domain(A::LinearOperatorAdjoint) = Range(A.inner)
Range(A::LinearOperatorAdjoint) = Domain(A.inner)

id(A::LinearOperatorAdjoint) = id(A.inner)
init(A::LinearOperatorAdjoint) = init(A.inner)
param(A::LinearOperatorAdjoint) = param(A.inner)
nparam(A::LinearOperatorAdjoint) = nparam(A.inner)
(A::LinearOperatorAdjoint)(pv::ParameterVector) = A.inner(pv)