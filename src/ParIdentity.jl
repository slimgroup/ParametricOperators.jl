export ParIdentity

struct ParIdentity{T} <: ParOperator{T,T,Linear,NonParametric,External}
    n::Int64
    id::ID
end

ParIdentity{T}(n::Int64) where {T} = ParIdentity{T}(n, uuid4(GLOBAL_RNG))

Domain(A::ParIdentity) = A.n
Range(A::ParIdentity) = A.n
id(A::ParIdentity) = A.id
adjoint(A::ParIdentity) = A

(::ParIdentity{T})(x::AbstractVecOrMat{T}) where {T} = x

function ∘(I1::ParIdentity{T}, I2::ParIdentity{T}) where {T}
    @assert I1.n == I2.n
    return I1
end

function kron(I1::ParIdentity{T}, I2::ParIdentity{T}) where {T}
    return ParIdentity{T}(I1.n*I2.n)
end

is_identity(A::ParOperator) = false
is_identity(A::ParIdentity) = true