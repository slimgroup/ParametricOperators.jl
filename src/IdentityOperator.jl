struct IdentityOperator{T} <: AbstractLinearOperator{T,T}
    n::Integer
    id::Any
end

IdentityOperator{T}(n::Integer) where {T} =
    IdentityOperator{T}(
        n,
        uuid4(GLOBAL_RNG)
    )

Domain(A::IdentityOperator) = A.n
Range(A::IdentityOperator) = A.n
param(A::IdentityOperator) = []
nparam(A::IdentityOperator) = 0

function init(A::IdentityOperator, pv::Optional{ParameterVector})
    if !isnothing(pv)
        pv[A.id] = []
    end
    return []
end

adjoint(A::IdentityOperator) = A
id(A::IdentityOperator) = A.id

*(A::IdentityOperator{T}, x::AbstractVector{T}) where {T} = x
*(A::IdentityOperator{T}, x::AbstractVecOrMat{T}) where {T} = x

(A::IdentityOperator{T})(θs::Any...) where {T} = A
(A::IdentityOperator{T})(pv::ParameterVector) where {T} = A
