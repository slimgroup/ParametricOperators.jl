export TaoMatrix

struct TaoMatrix{T} <: TaoOperator{T,T,Linear,Parametric,External}
    m::Int64
    n::Int64
    id::ID
end

TaoMatrix{T}(m::Int64, n::Int64) where {T} = TaoMatrix{T}(m, n, uuid4(GLOBAL_RNG))

Domain(A::TaoMatrix) = A.n
Range(A::TaoMatrix) = A.m
nparams(A::TaoMatrix) = A.m*A.n
id(A::TaoMatrix) = A.id
init(A::TaoMatrix{T}) where {T} = T(1/nparams(A))*rand(T, nparams(A))

(A::TaoParameterized{T,T,Linear,TaoMatrix{T},T,V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVecOrMat{T}} = reshape(A.θ, A.op.m, A.op.n)*x
(A::TaoAdjoint{T,T,NonParametric,TaoParameterized{T,T,Linear,TaoMatrix{T},T,V}})(y::Y) where 
    {T,V<:AbstractVector{T},Y<:AbstractVecOrMat{T}} = reshape(A.op.θ, A.op.op.m, A.op.op.n)'*y

function (K::TaoKron{T,T,T,T,T,T,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,TaoParameterized{T,T,Linear,TaoMatrix{T},T,V}})(x::X) where
    {T,F1,V<:AbstractVector{T},X<:AbstractVector{T}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = K.rhs*xr
    y2 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(y1))
    return vec(y2)
end

function (K::TaoKron{T,T,T,T,T,T,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,TaoIdentity{T},TaoParameterized{T,T,Linear,TaoMatrix{T},T,V}})(x::X) where
    {T,V<:AbstractVector{T},X<:AbstractVector{T}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = K.rhs*xr
    y2 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(y1))
    return vec(y2)
end