struct RestrictionOperator{T} <: Operator{T,T,Linear,NonParametric}
    shape::Vector{Int64}
    slices::Vector{Vector{UnitRange{Int64}}}
    id::Any
end

RestrictionOperator{T}(shape::Vector{Int64}, slices::Vector{Vector{UnitRange{Int64}}}) where {T} =
    RestrictionOperator{T}(
        shape,
        slices,
        uid()
    )

RestrictionOperator{T}(shape::Int64, slices::Vector{UnitRange{Int64}}) where {T} =
    RestrictionOperator{T}(
        [shape],
        [slices],
        uid()
    )

Domain(R::RestrictionOperator) = prod(R.shape)
Range(R::RestrictionOperator)  = sum(prod(r.stop-r.start+1 for r in v) for v in R.slices)
nparams(::RestrictionOperator) = 0
init(::RestrictionOperator)    = Vector{Vector{Nothing}}()
id(R::RestrictionOperator)     = R.id

(R::RestrictionOperator)(::Vector{<:AbstractArray}) = R

function (R::RestrictionOperator{T})(x::V) where {T<:Number,V<:AbstractVector{T}}
    xr = reshape(x, R.shape...)
    yr = zeros(T, Range(R))
    start = 1
    for sl in R.slices
        stop = start+prod(r.stop-r.start+1 for r in sl)-1
        yr[start:stop] = vec(xr[sl...])
        start = stop+1
    end
    return vec(yr)
end

function (A::Adjoint{T,T,NonParametric,RestrictionOperator{T}})(y::V) where {T<:Number,V<:AbstractVector{T}}
    R = A.op
    start = 1
    xr = zeros(T, R.shape...)
    for sl in R.slices
        sls = [r.stop-r.start+1 for r in sl]
        stop = start+prod(sls)-1
        xr[sl...] .+= reshape(y[start:stop], sls...)
        start = stop+1
    end
    return vec(xr)
end