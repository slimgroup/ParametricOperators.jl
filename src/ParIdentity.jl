export ParIdentity

"""
Identity operator.
"""
struct ParIdentity{T} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int
    ParIdentity(T, n) = new{T}(n)
    ParIdentity(n) = new{Float64}(n)
end

Domain(A::ParIdentity) = A.n
Range(A::ParIdentity) = A.n
adjoint(A::ParIdentity) = A
(A::ParIdentity{T})(x::X) where {T,X<:AbstractVector{T}} = x
(A::ParIdentity{T})(x::X) where {T,X<:AbstractMatrix{T}} = x

(A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}})(x::X) where {T,X<:AbstractVector{T}} = x
(A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}})(x::X) where {T,X<:AbstractMatrix{T}} = x
*(x::X, A::ParDistributed{T,T,Linear,NonParametric,ParIdentity{T}}) where {T,X<:AbstractMatrix{T}} = x
