export ParIdentity

struct ParIdentity{T} <: ParLinearOperator{T,T,NonParametric,FirstOrder}
    n::Int64
    ParIdentity(n) = new{Float64}(n)
    ParIdentity(T, n) = new{T}(n)
end

Domain(A::ParIdentity) = A.n
Range(A::ParIdentity) = A.n
id(A::ParIdentity) = "identity_$(A.n)"
adjoint(A::ParIdentity) = A

(::ParIdentity{T})(x::X) where {T,X<:AbstractVector{T}} = x
(::ParIdentity{T})(x::X) where {T,X<:AbstractMatrix{T}} = x