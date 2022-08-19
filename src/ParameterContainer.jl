import Base.+
import Base.-
import Base.*
import Base./
import Base.getindex
import Base.setindex!
import Base.zero
import Base.similar

using ChainRulesCore
using OrderedCollections

"""
Data structure for holding operator parameters. Serializes each type into
a different vector, which can be accessed via `container.data[T]` for some
type `T`.
"""
struct ParameterContainer
    type_map::OrderedDict{Any,Vector{<:Type}}
    range_map::OrderedDict{Any,Vector{UnitRange{Int64}}}
    shape_map::OrderedDict{Any,Vector{Shape}}
    data::OrderedDict{Type{<:Number},Vector{<:Number}}
end

"""
Default ParameterContainer constructor.
"""
ParameterContainer() = ParameterContainer(
    OrderedDict{Any,Vector{<:Type}}(),
    OrderedDict{Any,UnitRange{Int64}}(),
    OrderedDict{Any,Shape}(),
    OrderedDict{Type{<:Number},Vector{<:Number}}()
)

"""
zero ParameterContainer constructor.
"""
zero(pc::ParameterContainer) = ParameterContainer(
    pc.type_map,
    pc.range_map,
    pc.shape_map,
    OrderedDict{Type{<:Number},Vector{<:Number}}(
        T=>zero(v) for (T, v) in pc.data
    )
)

"""
Get the parameter(s) associated with a given id.
"""
function Base.getindex(pc::ParameterContainer, id::Any)
    θs = Vector{Array{<:Number}}()
    for (T, r, s) in zip(pc.type_map[id], pc.range_map[id], pc.shape_map[id])
        push!(θs, reshape(pc.data[T][r], s...))
    end
    return θs
end

"""
Set the parameter(s) associated with a given id.
"""
function Base.setindex!(pc::ParameterContainer, θ::A, id::Any) where {T<:Number,A<:AbstractArray{T}}
    if !haskey(pc.data, T)
        pc.data[T] = Vector{T}()
    end
    start = length(pc.data[T])+1
    append!(pc.data[T], vec(θ))
    stop = length(pc.data[T])
    pc.type_map[id] = [T]
    pc.range_map[id] = [start:stop]
    pc.shape_map[id] = [size(θ)]
end

"""
rrule for getting the parameter(s) associated with a given id.
"""
function ChainRulesCore.rrule(::typeof(getindex), pc::ParameterContainer, id::Any)
    θs = getindex(pc, id)
    function pc_getindex_pullback(∇θs)
        function pc_getindex_add!(Δpc)
            for (∇θ, T, r) in zip(∇θs, Δpc.type_map[id], Δpc.range_map[id])
                Δpc.data[T][r] += vec(∇θ)
            end
            return Δpc
        end

        ∇getindex = NoTangent()
        ∇pc = InplaceableThunk(
            pc_getindex_add!,
            @thunk(pc_getindex_add!(zero(pc)))
        )
        ∇id = NoTangent()

        return ∇getindex, ∇pc, ∇id
    end

    return θs, pc_getindex_pullback
end

# Define translation/scaling for parameter containers
for op in [:+, :-]
    @eval begin
        function Base.$op(lhs::ParameterContainer, rhs::ParameterContainer)
            return ParameterContainer(
                lhs.type_map,
                lhs.range_map,
                lhs.shape_map,
                OrderedDict{Type{<:Number},Vector{<:Number}}(
                    T => Base.$op(lhs.data[T], rhs.data[T]) for T in keys(lhs.data)
                )
            )
        end
    end
end

for op in [:+, :-, :*, :/]
    @eval begin
        function Base.$op(α::T, pc::ParameterContainer) where {T<:Number}
            return ParameterContainer(
                pc.type_map,
                pc.range_map,
                pc.shape_map,
                OrderedDict{Type{<:Number},Vector{<:Number}}(
                    D => Base.$op.(α, v) for (D, v) in pc.data
                )
            )
        end

        Base.$op(pc::ParameterContainer, α::T) where {T<:Number} = Base.$op(α, pc)
    end
end