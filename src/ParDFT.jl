export ParDFT, ParDFTN, ParDRFT

struct ParDFT{T<:Complex} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int64
    id::ID
end

ParDFT{T}(n::Int64) where {T<:Complex} = ParDFT{T}(n, uuid4(GLOBAL_RNG))

Domain(F::ParDFT) = F.n
Range(F::ParDFT) = F.n
id(F::ParDFT) = F.id

(F::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractVector{T}} = fft(x)/T(sqrt(F.n))
(F::ParAdjoint{T,T,NonParametric,ParDFT{T}})(y::Y) where {T<:Complex,Y<:AbstractVector{T}} = ifft(y)*T(sqrt(F.op.n))

struct ParDFTN{T<:Complex,N,M} <: ParLinearOperator{T,T,NonParametric,External}
    shape::NTuple{N,Int64}
    dims::NTuple{M,Int64}
    n::Int64
    scale::T
    id::ID
end

ParDFTN{T}(shape::NTuple{N,Int64}) where {T<:Complex,N} =
    ParDFTN{T,N,N}(
        shape,
        Tuple(1:N),
        prod(shape),
        T(sqrt(prod(shape))),
        uuid4(GLOBAL_RNG)
    )

ParDFTN{T}(shape::NTuple{N,Int64}, dims::NTuple{M,Int64}) where {T<:Complex,N,M} =
    ParDFTN{T,N,M}(
        shape,
        dims,
        prod(shape),
        sqrt(prod([shape[d] for d in dims])),
        uuid4(GLOBAL_RNG)
    )

Domain(F::ParDFTN) = F.n
Range(F::ParDFTN) = F.n
id(F::ParDFTN) = F.id

function (F::ParDFTN{T,M,N})(x::X) where {T<:Complex,M,N,X<:AbstractVector{T}}
    y = reshape(x, F.shape)
    y = fft(y, F.dims)/F.scale
    return vec(y)
end

function (A::ParAdjoint{T,T,NonParametric,ParDFTN{T,M,N}})(y::Y) where {T<:Complex,M,N,Y<:AbstractVector{T}}
    F = A.op
    x = reshape(y, F.shape)
    x = ifft(x, F.dims)*F.scale
    return vec(x)
end

kron(F1::ParDFT{T}, F2::ParDFT{T}) where {T<:Complex} = ParDFTN{T}((F2.n, F1.n))
kron(F1::ParDFT{T}, F2::ParDFTN{T,M,N}) where {T<:Complex,M,N} = ParDFTN{T}((F2.shape..., F1.n), (F2.dims..., N+1))
kron(F1::ParDFTN{T,M,N}, F2::ParDFT{T}) where {T<:Complex,M,N} = ParDFTN{T}((F2.n, F1.shape...), (1, (F1.dims.+1)...))
kron(F1::ParDFTN{T,K,L}, F2::ParDFTN{T,M,N}) where {T<:Complex,K,L,M,N} = ParDFTN{T}((F2.shape..., F1.shape...), (F2.dims..., (F1.dims.+N)...))

struct ParDRFT{T<:Real} <: ParOperator{T,Complex{T},Linear,NonParametric,External}
    m::Int64
    n::Int64
    id::ID
end

ParDRFT{T}(n::Int64) where {T<:Real} = ParDRFT{T}(n÷2+1, n, uuid4(GLOBAL_RNG))

Domain(F::ParDRFT) = F.n
Range(F::ParDRFT) = F.m
id(F::ParDRFT) = F.id

function (F::ParDRFT{T})(x::X) where {T<:Real,X<:AbstractVector{T}}
    y = fft(x)/Complex{T}(sqrt(F.n))
    return y[1:F.m]
end

function (A::ParAdjoint{T,Complex{T},NonParametric,ParDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractVector{Complex{T}}}
    F = A.op
    k = iseven(F.m) ? F.m : F.m-1
    x = vcat(y, conj(y[k:-1:2]))
    return real(ifft(x)*T(sqrt(F.n)))
end