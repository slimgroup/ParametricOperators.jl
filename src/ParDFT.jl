export ParDFT

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
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(x::X) where {T<:Complex,X<:AbstractVector{T}} = ifft(x).*T(sqrt(A.n))
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(x::X) where {T<:Complex,X<:AbstractMatrix{T}} = ifft(x, 1).*T(sqrt(A.n))