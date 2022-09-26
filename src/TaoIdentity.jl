export TaoIdentity

struct TaoIdentity{T} <: TaoOperator{T,T,Linear,NonParametric,External}
    n::Int64
    id::ID
end

TaoIdentity{T}(n::Int64) where {T} = TaoIdentity{T}(n, uuid4(GLOBAL_RNG))

Domain(A::TaoIdentity) = A.n
Range(A::TaoIdentity) = A.n
id(A::TaoIdentity) = A.id

adjoint(A::TaoIdentity) = A
(::TaoIdentity{T})(x::AbstractVecOrMat{T}) where {T} = x

function (K::TaoKron{D,D,D,R,D,R,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,TaoIdentity{D},F2})(x::X) where
    {D,R,F2,X<:AbstractVector{D}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y = mapreduce(c -> K.rhs*c, hcat, eachcol(xr))
    return vec(y)
end

function (K::TaoKron{D,D,D,R,R,D,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,TaoIdentity{D}})(x::X) where
    {D,R,F1,X<:AbstractVector{D}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(xr))
    return vec(y)
end

kron(A1::TaoIdentity{T}, A2::TaoIdentity{T}) where {T} = TaoIdentity{T}(A1.n*A2.n)