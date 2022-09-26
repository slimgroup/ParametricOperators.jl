export TaoDRFT

struct TaoDRFT{T<:Real} <: TaoOperator{T,Complex{T},Linear,NonParametric,External}
    n::Int64
    id::Any
end

TaoDRFT{T}(n::Int64) where {T<:Real} = TaoDRFT{T}(n, uuid4(GLOBAL_RNG))

Domain(F::TaoDRFT) = F.n
Range(F::TaoDRFT) = F.n÷2+1
id(F::TaoDRFT) = F.id

function (F::TaoDRFT{T})(x::X) where {T<:Real,X<:AbstractVector{T}}
    y = fft(x)./T(sqrt(F.n))
    return y[1:F.n÷2+1]
end

function (A::TaoAdjoint{T,Complex{T},NonParametric,TaoDRFT{T}})(y::Y) where {T<:Real,Y<:AbstractVector{Complex{T}}}
    F = A.op
    xl = ifft(y).*T(sqrt(F.n))
    x = zeros(Complex{T}, F.n)
    x[1:F.n÷2+1] .= xl
    x[end:-1:F.n÷2+2] .= conj(xl[2:(F.n-1)÷2+1])
    return real(x)
end