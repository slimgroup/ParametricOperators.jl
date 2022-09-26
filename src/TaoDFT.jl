export TaoDFT, TaoDFTN

struct TaoDFT{T<:Complex} <: TaoOperator{T,T,Linear,NonParametric,External}
    n::Int64
    id::ID
end

TaoDFT{T}(n::Int64) where {T} = TaoDFT{T}(n, uuid4(GLOBAL_RNG))

Domain(F::TaoDFT) = F.n
Range(F::TaoDFT) = F.n
id(F::TaoDFT) = F.id

(F::TaoDFT{T})(x::X) where {T,X<:AbstractVecOrMat{T}} = fft(x)./T(sqrt(F.n))
(F::TaoAdjoint{T,T,NonParametric,TaoDFT{T}})(y::Y) where {T,Y<:AbstractVecOrMat{T}} = ifft(y).*T(sqrt(F.op.n))

struct TaoDFTN{T<:Complex,N,K} <: TaoOperator{T,T,Linear,NonParametric,External}
    shape::NTuple{N,Int64}
    dims::NTuple{K,Int64}
    id::ID
end

function TaoDFTN{T}(shape::NTuple{N,Int64}) where {T,N}
    dims = Tuple(1:N)
    return TaoDFTN{T,N,N}(shape, dims, uuid4(GLOBAL_RNG))
end

function TaoDFTN{T}(shape::NTuple{N,Int64}, dims::NTuple{K,Int64}) where {T,N,K}
    return TaoDFTN{T,N,K}(shape, dims, uuid4(GLOBAL_RNG))
end

Domain(F::TaoDFTN) = prod(F.shape)
Range(F::TaoDFTN) = prod(F.shape)
id(F::TaoDFTN) = F.id

scale(F::TaoDFTN) = sqrt(prod([F.shape[d] for d in F.dims]))

(F::TaoDFTN{T,N,K})(x::X) where {T,N,K,X<:AbstractVector{T}} = vec(fft(reshape(x, F.shape), F.dims)./T(scale(F)))
(F::TaoAdjoint{T,T,NonParametric,TaoDFTN{T,N,K}})(y::Y) where {T,Y<:AbstractVector{T},N,K} = vec(ifft(reshape(y, F.op.shape), F.op.dims).*T(scale(F.op)))

kron(F1::TaoDFT{T}, F2::TaoDFT{T}) where {T<:Complex} = TaoDFTN{T}((F2.n, F1.n))
kron(F1::TaoDFT{T}, F2::TaoDFTN{T,N,K}) where {T<:Complex,N,K} = TaoDFTN{T}((F2.shape..., F1.n), (F2.dims..., N+1))
kron(F1::TaoDFTN{T,N,K}, F2::TaoDFT{T}) where {T<:Complex,N,K} = TaoDFTN{T}((F2.n, F1.shape...), (1, [d+1 for d in F1.dims]...))
kron(F1::TaoDFTN{T,N,K}, F2::TaoDFTN{T,M,J}) where {T<:Complex,N,M,K,J} = TaoDFTN{T}((F2.shape..., F1.shape...), (F2.dims..., [d+M for d in F1.dims]...))

# Faster to dispatch kernel on the entire tensor than to repeatedly take fourier transforms
kron(lhs::F1, rhs::F2) where {T<:Complex,N,K,P,F1<:TaoDFTN{T,N,K},F2<:TaoOperator{T,T,Linear,P,<:NodeType}} =
    TaoDFTN{T}((Domain(rhs), lhs.shape...), Tuple([d+1 for d in lhs.dims])) ∘
    (TaoIdentity{T}(Domain(lhs)) ⊗ rhs)
    