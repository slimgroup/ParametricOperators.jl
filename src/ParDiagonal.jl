export ParDiagonal

struct ParDiagonal{T} <: ParLinearOperator{T,T,Parametric,FirstOrder}
    n::Int64
    id::ID
    ParDiagonal(n) = new{Float64}(n, uuid4(GLOBAL_RNG))
    ParDiagonal(T, n) = new{T}(n, uuid4(GLOBAL_RNG))
end

Domain(A::ParDiagonal) = A.n
Range(A::ParDiagonal) = A.n
nparams(A::ParDiagonal) = A.n
init(A::ParDiagonal{T}) where {T} = randn(T, A.n)

function (A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVector{T}}
    return A.θ.*x
end

function (A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractMatrix{T}}
    return A.θ.*x
end

function (A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParDiagonal{T},V}})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVector{T}}
    return conj(A.op.θ).*x
end

function (A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParDiagonal{T},V}})(x::X) where {T,V<:AbstractVector{T},X<:AbstractMatrix{T}}
    return conj(A.op.θ).*x
end