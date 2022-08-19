using FFTW

struct DFTOperator{T} <: AbstractLinearOperator{T,T}
    shape::Shape
    dims::Shape
    id::Any
end

function DFTOperator{T}(shape::Shape, dims::Optional{Shape} = nothing) where {T}
    dims_out = isnothing(dims) ? collect(1:length(shape)) : dims
    return DFTOperator{T}(
        shape,
        dims_out,
        uid()
    )
end

# DFT operator impls
Domain(F::DFTOperator) = prod(F.shape)
Range(F::DFTOperator) = prod(F.shape)
params(F::DFTOperator) = []
nparams(F::DFTOperator) = 0
id(F::DFTOperator) = F.id
init(::DFTOperator, ::Optional{ParameterContainer} = nothing) = []

function *(F::DFTOperator{T}, x::V) where {T<:Number,V<:AbstractVector{T}}
    rx = reshape(x, F.shape...)
    ry = fft(rx, F.dims)/T(sqrt(prod(map(i -> F.shape[i], F.dims))))
    y = vec(ry)
    return y
end

function *(F::LinearOperatorAdjoint{T,T,A}, y::V) where {T<:Number,A<:DFTOperator{T},V<:AbstractVector{T}}
    ry = reshape(y, F.inner.shape...)
    rx = ifft(ry, F.inner.dims)*T(sqrt(prod(map(i -> F.inner.shape[i], F.inner.dims))))
    x = vec(rx)
    return x
end

(F::DFTOperator)(::Vararg{<:AbstractArray}) = F
(F::DFTOperator)(::ParameterContainer) = F