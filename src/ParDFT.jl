export ParDFT, ParDRFT, ParDFTN

struct ParDFT{T<:Complex} <: ParLinearOperator{T,T,NonParametric,FirstOrder}
    n::Int64
    id::ID
    ParDFT(n) = new{ComplexF64}(n, uuid4(GLOBAL_RNG))
    ParDFT(T, n) = new{T}(n, uuid4(GLOBAL_RNG))
end

Domain(A::ParDFT) = A.n
Range(A::ParDFT) = A.n
id(A::ParDFT) = A.id

(A::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractVector{T}} = fft(x)./T(sqrt(A.n))
(A::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractMatrix{T}} = fft(x, 1)./T(sqrt(A.n))
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(y::Y) where {T<:Complex,Y<:AbstractVector{T}} = ifft(y).*T(sqrt(A.n))
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(y::Y) where {T<:Complex,Y<:AbstractMatrix{T}} = ifft(y, 1).*T(sqrt(A.n))

struct ParDRFT{T<:Real} <: ParLinearOperator{T,Complex{T},NonParametric,FirstOrder}
    m::Int64
    n::Int64
    id::ID
    ParDRFT(n) = new{Float64}(n÷2+1, n, uuid4(GLOBAL_RNG))
    ParDRFT(T, n) = new{T}(n÷2+1, n, uuid4(GLOBAL_RNG))
end

Domain(A::ParDRFT) = A.n
Range(A::ParDRFT) = A.m
id(A::ParDRFT) = A.id

function (A::ParDRFT{T})(x::X) where {T<:Real,X<:AbstractVector{T}}
    y = fft(x)./Complex{T}(sqrt(A.n))
    return y[1:A.m]
end

function (A::ParDRFT{T})(x::X) where {T<:Real,X<:AbstractMatrix{T}}
    y = fft(x, 1)./Complex{T}(sqrt(A.n))
    return y[1:A.m,:]
end

function (A::ParAdjoint{T,Complex{T},NonParametric,ParDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractVector{Complex{T}}}
    k = iseven(A.op.m) ? A.op.m : A.op.m-1
    x = vcat(y, conj(y[k:-1:2]))
    return real(ifft(x).*T(sqrt(A.op.n)))
end

function (A::ParAdjoint{T,Complex{T},NonParametric,ParDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractMatrix{Complex{T}}}
    k = iseven(A.op.m) ? A.op.m : A.op.m-1
    x = vcat(y, conj(y[k:-1:2,:]))
    return real(ifft(x, 1).*T(sqrt(A.op.n)))
end

struct ParDFTN{T<:Complex,N,K} <: ParSeparableOperator{T,T,NonParametric,FirstOrder}
    n::Int64
    shape::NTuple{N, Int64}
    dims::NTuple{K, Int64}
    id::ID
    function ParDFTN(shape; dims=nothing)
        N = length(shape)
        dims = isnothing(dims) ? Tuple(1:N) : dims
        K = length(dims)
        return new{ComplexF64,N,K}(prod(shape), shape, dims, uuid4(GLOBAL_RNG))
    end
    function ParDFTN(T, shape; dims=nothing)
        N = length(shape)
        dims = isnothing(dims) ? Tuple(1:N) : dims
        K = length(dims)
        return new{T,N,K}(prod(shape), shape, dims, uuid4(GLOBAL_RNG))
    end
end

Domain(A::ParDFTN) = A.n
Range(A::ParDFTN) = A.n
id(A::ParDFTN) = A.id
decomposition(A::ParDFTN{T,N,K}) where {T,N,K} =
    collect(map(j -> j ∈ A.dims ? ParDFT(T, A.shape[j]) : ParIdentity(T, A.shape[j]), 1:N))

function (A::ParDFTN{T,N,K})(x::X) where {T<:Complex,N,K,X<:AbstractVector{T}}
    y = reshape(x, A.shape)
    y = fft(y, A.dims)./T(sqrt(A.n))
    return vec(y)
end

function (A::ParDFTN{T,N,K})(x::X) where {T<:Complex,N,K,X<:AbstractMatrix{T}}
    _, nc = size(x)
    y = reshape(x, A.shape..., nc)
    y = fft(y, A.dims)./T(sqrt(A.n))
    return reshape(y, Range(A), nc)
end

function (A::ParAdjoint{T,T,NonParametric,ParDFTN{T,N,K}})(y::Y) where {T<:Complex,N,K,Y<:AbstractVector{T}}
    x = reshape(y, A.shape)
    x = ifft(x, A.dims).*T(sqrt(A.n))
    return vec(x)
end

function (A::ParAdjoint{T,T,NonParametric,ParDFTN{T,N,K}})(y::Y) where {T<:Complex,N,K,Y<:AbstractMatrix{T}}
    _, nc = size(y)
    x = reshape(y, A.op.shape..., nc)
    x = ifft(x, A.op.dims).*T(sqrt(A.op.n))
    return reshape(x, Domain(A.op), nc)
end

kron(A::ParDFT{T}, B::ParDFT{T}) where {T} = ParDFTN((B.n, A.n))
kron(A::ParDFT{T}, B::ParDFTN{T,N,K}) where {T,N,K} =
    ParDFTN((B.shape..., A.n); dims=Tuple(vcat(collect(B.dims), N+1)))
kron(A::ParDFTN{T,N,K}, B::ParDFT{T}) where {T,N,K} =
    ParDFTN((B.n, A.shape...); dims=Tuple(vcat(1, collect(A.dims) .+1 )))
kron(A::ParDFTN{T,N1,K1}, B::ParDFTN{T,N2,K2}) where {T,N1,N2,K1,K2} =
    ParDFTN((B.shape..., A.shape...); dims=Tuple(vcat(collect(B.dims), collect(A.dims) .+ N2)))