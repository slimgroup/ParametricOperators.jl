export ParDFT, ParDFTN

struct ParDFT{T<:Complex} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int64
    id::ID
    ParDFT(n; T = ComplexF64) = new{T}(n, uuid4(GLOBAL_RNG))
end

Domain(A::ParDFT) = A.n
Range(A::ParDFT) = A.n
id(A::ParDFT) = A.id

(A::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractVector{T}} = fft(x)./T(sqrt(A.n))
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(x::X) where {T<:Complex,X<:AbstractVector{T}} = ifft(x).*T(sqrt(A.op.n))

struct ParDFTN{N,K,T<:Complex} <: ParLinearOperator{T,T,NonParametric,External}
    shape::NTuple{N, Int64}
    dims::NTuple{K, Int64}
    n::Int64
    id::ID
    ParDFTN(ns...; T = ComplexF64) = new{length(ns),length(ns),T}(
        ns, 
        Tuple(1:length(ns)),
        prod(ns),
        uuid4(GLOBAL_RNG)
    )
end

Domain(A::ParDFTN) = A.n
Range(A::ParDFTN) = A.n
id(A::ParDFTN) = A.id

function (A::ParDFTN{N,K,T})(x::X) where {N,K,T<:Complex,X<:AbstractVector{T}}
    xr = reshape(x, A.shape)
    yr = fft(xr, A.dims)./T(sqrt(A.n))
    return vec(yr)
end

function (A::ParAdjoint{T,T,NonParametric,ParDFTN{N,K,T}})(y::Y) where {N,K,T<:Complex,Y<:AbstractVector{T}}
    yr = reshape(y, A.op.shape)
    xr = ifft(yr, A.op.dims).*T(sqrt(A.op.n))
    return vec(xr)
end

optimize(A::ParKron{T,T,NonParametric,ParDFT{T},ParDFT{T}}) where {T<:Complex} = ParDFTN(A.rhs.n, A.lhs.n; T=T)
optimize(A::ParKron{T,T,NonParametric,ParDFT{T},ParDFTN{N,K,T}}) where {N,K,T<:Complex} = ParDFTN(A.rhs.shape..., A.lhs.n; T=T)
optimize(A::ParKron{T,T,NonParametric,ParDFTN{N,K,T},ParDFT{T}}) where {N,K,T<:Complex} = ParDFTN(A.rhs.n, A.lhs.shape...; T=T)
optimize(A::ParKron{T,T,NonParametric,ParDFTN{M,K,T},ParDFTN{N,L,T}}) where {M,K,N,L,T<:Complex} = ParDFTN(A.rhs.shape..., A.lhs.shape...; T=T)