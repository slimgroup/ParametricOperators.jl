export ParMatrix

struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,FirstOrder}
    m::Int64
    n::Int64
    id::ID
    ParMatrix(m, n) = new{Float64}(m, n, uuid4(GLOBAL_RNG))
    ParMatrix(T, m, n) = new{T}(m, n, uuid4(GLOBAL_RNG))
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m
nparams(A::ParMatrix) = A.m*A.n
init(A::ParMatrix{T}) where {T} = T(1/nparams(A)).*rand(T, nparams(A))

function (A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVector{T}}
    return reshape(A.θ, A.op.m, A.op.n)*x
end

function (A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractMatrix{T}}
    return reshape(A.θ, A.op.m, A.op.n)*x
end

function (A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParMatrix{T},V}})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVector{T}}
    return reshape(A.op.θ, A.op.op.m, A.op.op.n)'*x
end

function (A::ParAdjoint{T,T,Parameterized,ParParameterized{T,T,Linear,ParMatrix{T},V}})(x::X) where {T,V<:AbstractVector{T},X<:AbstractMatrix{T}}
    return reshape(A.op.θ, A.op.op.m, A.op.op.n)'*x
end