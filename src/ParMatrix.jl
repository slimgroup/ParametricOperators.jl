export ParMatrix

struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int64
    n::Int64
    id::ID
end

ParMatrix{T}(m::Int64, n::Int64) where {T} = ParMatrix{T}(m, n, uuid4(GLOBAL_RNG))

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m
id(A::ParMatrix) = A.id
nparams(A::ParMatrix) = A.m*A.n
init(A::ParMatrix{T}) where {T} = T(1/nparams(A))*rand(T, nparams(A))

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where
    {T,V<:AbstractVector{T},X<:AbstractVector{T}} = reshape(A.θ, A.op.m, A.op.n)*x

(A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParMatrix{T},V}})(y::Y) where
    {T,V<:AbstractVector{T},Y<:AbstractVector{T}} = reshape(A.op.θ, A.op.op.m, A.op.op.n)'*y

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where
    {T,V<:AbstractVector{T},X<:AbstractMatrix{T}} = reshape(A.θ, A.op.m, A.op.n)*x

(A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParMatrix{T},V}})(y::Y) where
    {T,V<:AbstractVector{T},Y<:AbstractMatrix{T}} = reshape(A.op.θ, A.op.op.m, A.op.op.n)'*y

function (K::ParKron{T,T,Parameterized,Tuple{ParIdentity{T},ParMatrix{T}}})(x::X) where {T,X<:AbstractVector{T}}
    y = reshape(x, K.shape_in...)
    y = K.ops[1]*y
    return vec(y)
end

# TODO: Types
kron(A::ParLinearOperator, B::ParMatrix{T}) where {T} = ParKron(A, ParIdentity{DDT(A)}(Range(B)))*ParKron(ParIdentity{T}(Domain(A)), B)