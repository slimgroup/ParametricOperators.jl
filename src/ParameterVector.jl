import Base.setindex!
import Base.getindex
import Base.zero
import Base.+
import Base.-
import Base.*

using ChainRulesCore

struct ParameterVector{T}
    data::Vector{T}
    range_map::Dict{Any,UnitRange}
    shape_map::Dict{Any,Shape}
end

ParameterVector{T}() where {T} = ParameterVector{T}(
    Vector{T}(),
    Dict{Any,UnitRange}(),
    Dict{Any,Shape}()
)

function zero(pv::ParameterVector{T}) where {T}
    return ParameterVector{T}(
        zero(pv.data),
        pv.range_map,
        pv.shape_map
    )
end

function setindex!(pv::ParameterVector{T}, θ::Array{T}, id::Any) where {T}
    if haskey(pv.range_map, id)
        s = pv.shape_map[id]
        @assert s == size(θ)
        pv.data[pv.range_map[id]] = vec(θ)
    else
        shape = size(θ)
        start = length(pv.data)+1
        stop = start + prod(shape) - 1
        append!(pv.data, vec(θ))
        pv.range_map[id] = start:stop
        pv.shape_map[id] = shape
    end
end

function getindex(pv::ParameterVector{T}, id::Any) where {T}
    return reshape(pv.data[pv.range_map[id]], pv.shape_map[id])
end

function ChainRulesCore.rrule(::typeof(getindex), pv::ParameterVector{T}, id::Any) where {T}
    θ = getindex(pv, id)
    function pv_getindex_pullback(θ̄)
        function pv_getindex_add!(Δpv)
            Δpv.data[Δpv.range_map[id]] += vec(θ̄)
            return Δpv
        end

        ḡetindex = NoTangent()
        p̄v = InplaceableThunk(
            pv_getindex_add!,
            @thunk(pv_getindex_add!(zero(pv)))
        )
        īd = NoTangent()
        return ḡetindex, p̄v, īd
    end

    return θ, pv_getindex_pullback
end

function +(lhs::ParameterVector{T}, rhs::ParameterVector{T}) where {T}
    out = zero(lhs)
    for (id, range_rhs) in rhs.range_map
        range_lhs = out.range_map[id]
        out.data[range_lhs] = lhs.data[range_lhs] + rhs.data[range_rhs]
    end
    return out
end

function -(lhs::ParameterVector{T}, rhs::ParameterVector{T}) where {T}
    out = zero(lhs)
    for (id, range_rhs) in rhs.range_map
        range_lhs = out.range_map[id]
        out.data[range_lhs] = lhs.data[range_lhs] - rhs.data[range_rhs]
    end
    return out
end

*(α::T, pv::ParameterVector{T}) where {T} = ParameterVector{T}(
    α*pv.data,
    pv.range_map,
    pv.shape_map
)