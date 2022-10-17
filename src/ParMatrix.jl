export ParMatrix

struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int64
    n::Int64
    id::ID
    ParMatrix(m, n; T = Float64) = new{T}(m, n, uuid4(GLOBAL_RNG))
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m
nparams(A::ParMatrix) = A.m*A.n
init(A::ParMatrix{T}) where {T} = T(1/nparams(A)) .* rand(T, nparams(A))
id(A::ParMatrix) = A.id

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where
    {T,X<:AbstractVector{T},V<:AbstractVector{T}} = reshape(A.θ, A.op.m, A.op.n)*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where
    {T,X<:AbstractVector{T},V<:AbstractVector{T}} = reshape(A.θ, A.op.op.m, A.op.op.n)'*x