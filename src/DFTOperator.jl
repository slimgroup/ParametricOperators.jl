struct DFTOperator{T<:Number} <: Operator{T,T,Linear,NonParametric}
    shape::Vector{Int64}
    dims::Vector{Int64}
    id::Any
end

DFTOperator{T}(shape::Vector{Int64}, dims::Vector{Int64}) where {T} =
    DFTOperator{T}(
        shape,
        dims,
        uid()
    )

DFTOperator{T}(shape::Vector{Int64}) where {T} =
    DFTOperator{T}(
        shape,
        collect(1:length(shape)),
        uid()
    )

Domain(F::DFTOperator)  = prod(F.shape)
Range(F::DFTOperator)   = prod(F.shape)
nparams(F::DFTOperator) = 0
init(::DFTOperator)     = Vector{Vector{Nothing}}()
id(F::DFTOperator)      = F.id

(F::DFTOperator)(::Vector{<:AbstractArray}) = F

function (F::DFTOperator{T})(x::V) where {T<:Number,V<:AbstractVector{T}}
    scale = sqrt(prod([F.shape[d] for d in F.dims]))
    xr = reshape(x, F.shape...)
    yr = fft(xr, F.dims)/T(scale)
    return vec(yr)
end

function (A::Adjoint{T,T,NonParametric,DFTOperator{T}})(y::V) where {T<:Number,V<:AbstractVector{T}}
    F = A.op
    scale = sqrt(prod([F.shape[d] for d in F.dims]))
    yr = reshape(y, F.shape...)
    xr = ifft(yr, F.dims)*T(scale)
    return vec(xr)
end