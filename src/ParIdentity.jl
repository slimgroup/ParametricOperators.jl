export ParIdentity

struct ParIdentity{T} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int64
    ParIdentity(n; T = Float64) = new{T}(n)
end

Domain(A::ParIdentity) = A.n
Range(A::ParIdentity) = A.n
id(A::ParIdentity) = "I$(A.n)"
adjoint(A::ParIdentity) = A

(::ParIdentity)(x::AbstractVecOrMat) = x