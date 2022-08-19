using FFTW

struct DRFTOperator{D,R} <: AbstractLinearOperator{D,R}
    shape::Shape
    dims::Shape
    id::Any
end

function DRFTOperator{D,R}(shape::Shape, dims::Optional{Shape} = nothing) where {D,R}
    dims_out = isnothing(dims) ? collect(1:length(shape)) : dims
    return DRFTOperator{D,R}(
        shape,
        dims_out,
        uid()
    )
end

# DFT operator impls
Domain(F::DRFTOperator) = prod(F.shape)
Range(F::DRFTOperator) = prod([(i == F.dims[1]) ? (s÷2+1) : s for (i, s) in enumerate(F.shape)])
params(F::DRFTOperator) = []
nparams(F::DRFTOperator) = 0
id(F::DRFTOperator) = F.id
init(::DRFTOperator, ::Optional{ParameterContainer} = nothing) = []

function retype_domain(F::DRFTOperator{D,R}, t::T) where {D,R,T}
    return DRFTOperator{t,R}(
        F.shape,
        F.dims,
        F.id
    )
end

function *(F::DRFTOperator{D,R}, x::V) where {D<:Number,R<:Number,V<:AbstractVector{D}}
    rx = reshape(x, F.shape...)
    ry = fft(rx, F.dims)/R(sqrt(prod(map(i -> F.shape[i], F.dims))))
    ranges = [(i == F.dims[1]) ? (1:(s÷2+1)) : (1:s) for (i, s) in enumerate(F.shape)]
    y = vec(ry[ranges...])
    return y
end

function *(F::LinearOperatorAdjoint{D,R,A}, y::V) where {D,R,A<:DRFTOperator{D,R},V<:AbstractVector{R}}
    ranges = [(i == F.inner.dims[1]) ? (1:(s÷2+1)) : (1:s) for (i, s) in enumerate(F.inner.shape)]
    out_shape = [(i == F.inner.dims[1]) ? (s÷2+1) : s for (i, s) in enumerate(F.inner.shape)]
    ry = zeros(R, F.inner.shape)
    ry[ranges...] = reshape(y, out_shape...)
    rx = ifft(ry, F.inner.dims)*D(sqrt(prod(map(i -> F.inner.shape[i], F.inner.dims))))
    return vec(rx)
end

(F::DRFTOperator)(::Vararg{<:AbstractArray}) = F
(F::DRFTOperator)(::ParameterContainer) = F