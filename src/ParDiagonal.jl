export ParDiagonal

struct ParDiagonal{T} <: ParLinearOperator{T,T,Parametric,External}
    n::Int64
    id::ID
end

ParDiagonal{T}(n::Int64) where {T} = ParDiagonal{T}(n, uuid4(GLOBAL_RNG))

Domain(A::ParDiagonal) = A.n
Range(A::ParDiagonal) = A.n
id(A::ParDiagonal) = A.id
nparams(A::ParDiagonal) = A.n
init(A::ParDiagonal{T}) where {T} = ones(T, A.n)

(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where
    {T,V<:AbstractVector{T},X<:AbstractVector{T}} = A.θ.*x
(A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParDiagonal{T},V}})(y::Y) where
    {T,V<:AbstractVector{T},Y<:AbstractVector{T}} = conj(A.op.θ).*y