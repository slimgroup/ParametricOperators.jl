export ParDFT, ParDRFT

"""
Discrete Fourier transform operator. Implemented using FFT.
"""
struct ParDFT{T<:Complex} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int
    ParDFT(T, n) = new{T}(n)
    ParDFT(n) = new{ComplexF64}(n)
end

Domain(A::ParDFT) = A.n
Range(A::ParDFT) = A.n

(A::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractVector{T}} = fft(x)./sqrt(A.n)
(A::ParDFT{T})(x::X) where {T<:Complex,X<:AbstractMatrix{T}} = fft(x, 1)./sqrt(A.n)
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(x::X) where {T<:Complex,X<:AbstractVector{T}} = ifft(x).*sqrt(A.op.n)
(A::ParAdjoint{T,T,NonParametric,ParDFT{T}})(x::X) where {T<:Complex,X<:AbstractMatrix{T}} = ifft(x, 1).*sqrt(A.op.n)

"""
Discrete real-valued Fourier transform operator.
"""
struct ParDRFT{T<:Real} <: ParLinearOperator{T,Complex{T},NonParametric,External}
    n::Int
    ParDRFT(T, n) = new{T}(n)
    ParDRFT(n) = new{Float64}(n)
end

Domain(A::ParDRFT) = A.n
Range(A::ParDRFT) = div(A.n, 2) + 1

(A::ParDRFT{T})(x::X) where {T<:Real,X<:AbstractVector{T}} = (fft(x)./Complex{T}(sqrt(A.n)))[1:Range(A)]
(A::ParDRFT{T})(x::X) where {T<:Real,X<:AbstractMatrix{T}} = (fft(x, 1)./Complex{T}(sqrt(A.n)))[1:Range(A),:]

function (A::ParAdjoint{T,Complex{T},NonParametric,ParDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractVector{Complex{T}}}
    m = Range(A.op)
    k = iseven(m) ? m : m-1
    x = vcat(y, conj(y[k:-1:2]))
    return real(ifft(x).*T(sqrt(A.op.n)))
end

function (A::ParAdjoint{T,Complex{T},NonParametric,ParDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractMatrix{Complex{T}}}
    m = Range(A.op)
    k = iseven(m) ? m : m-1
    x = vcat(y, conj(y[k:-1:2,:]))
    return real(ifft(x).*T(sqrt(A.op.n)))
end
