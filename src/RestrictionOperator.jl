struct RestrictionOperator{T} <: AbstractLinearOperator{T,T}
    n::Integer
    slices::AbstractVector{<:AbstractRange}
    adjoint::Bool
    id::Any
end

RestrictionOperator{T}(n::Integer, slices::AbstractVector{<:AbstractRange}) where{T} =
    RestrictionOperator{T}(
        n,
        slices,
        false,
        uuid4(GLOBAL_RNG)
    )

Domain(R::RestrictionOperator) = R.adjoint ? sum(map(r -> length(r), R.slices)) : R.n
Range(R::RestrictionOperator) = R.adjoint ? R.n : sum(map(r -> length(r), R.slices))
param(R::RestrictionOperator) = []
nparam(R::RestrictionOperator) = 0

function init(R::RestrictionOperator{T}, pv::Optional{ParameterVector}) where {T}
    if !isnothing(pv)
        pv[R.id] = []
    end
    return []
end

adjoint(R::RestrictionOperator{T}) where {T} =
    RestrictionOperator{T}(
        R.n,
        R.slices,
        !R.adjoint,
        R.id
    )

id(R::RestrictionOperator) = R.id
function *(R::RestrictionOperator{T}, x::AbstractVector{T}) where {T<:Number}
    if R.adjoint
        y = zeros(T, Range(R))
        offset = 1
        for sl in R.slices
            y[sl] = x[offset:offset+length(sl)-1]
            offset += length(sl)
        end
        return y
    else
        return reduce(vcat, map(sl -> x[sl], R.slices))
    end
end

(R::RestrictionOperator{T})(θs::Any...) where {T} = R
(R::RestrictionOperator{T})(pv::ParameterVector) where {T} = R
