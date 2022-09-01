struct DRFTOperator{D<:Number,R<:Complex} <: Operator{D,R,Linear,NonParametric}
    shape::Vector{Int64}
    dims::Vector{Int64}
    id::Any
end

DRFTOperator{D,R}(shape::Vector{Int64}, dims::Vector{Int64}) where {D,R} =
    DRFTOperator{D,R}(
        shape,
        dims,
        uid()
    )

DRFTOperator{D,R}(shape::Vector{Int64}) where {D,R} =
    DRFTOperator{D,R}(
        shape,
        collect(1:length(shape)),
        uid()
    )

Domain(F::DRFTOperator)  = prod(F.shape)
Range(F::DRFTOperator)   = prod([i == F.dims[1] ? s÷2+1 : s for (i, s) in enumerate(F.shape)])
nparams(F::DRFTOperator) = 0
init(::DRFTOperator)     = Vector{Vector{Nothing}}()
id(F::DRFTOperator)      = F.id

(F::DRFTOperator)(::Vector{<:AbstractArray}) = F

function (F::DRFTOperator{D,R})(x::V) where {D<:Number,R<:Complex,V<:AbstractVector{D}}
    scale = sqrt(prod([F.shape[d] for d in F.dims]))
    xr = reshape(x, F.shape...)
    yr = rfft(xr, F.dims)/R(scale)
    return vec(yr)
end

function (A::Adjoint{D,R,NonParametric,DRFTOperator{D,R}})(y::V) where {D<:Number,R<:Complex,V<:AbstractVector{R}}
    F = A.op
    scale = sqrt(prod([F.shape[d] for d in F.dims]))
    out_shape = [i == F.dims[1] ? s÷2+1 : s for (i, s) in enumerate(F.shape)]
    yr = reshape(y, out_shape...)
    xr = D.(irfft(yr, F.shape[F.dims[1]], F.dims))*D(scale)
    return vec(xr)
end