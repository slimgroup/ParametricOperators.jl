"""
Package exception type.
"""
struct ParException <: Exception
    msg::String
end

"""
Option type. Contains T or nothing.
"""
const Option{T} = Union{T, Nothing}

"""
Given a vector of numbers, compute the ranges between the cumsum of those numbers.
"""
function cumranges(x::Vector{<:Integer})
    offsets = [0, cumsum(x[1:end-1])...]
    starts = offsets .+ 1
    stops = offsets .+ x
    return [start:stop for (start, stop) in zip(starts, stops)]
end

FLOAT_TYPES = [:Float16, :Float32, :Float64]
for i in 1:length(FLOAT_TYPES)
    for j in i:length(FLOAT_TYPES)
        @eval begin
            subset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{$(FLOAT_TYPES[j])}) = $(FLOAT_TYPES[i])
            subset_type(::Type{Complex{$(FLOAT_TYPES[i])}}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            subset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = $(FLOAT_TYPES[i])
            subset_type(::Type{Complex{$(FLOAT_TYPES[j])}}, ::Type{$(FLOAT_TYPES[i])}) = $(FLOAT_TYPES[i])
            superset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{$(FLOAT_TYPES[j])}) = $(FLOAT_TYPES[j])
            superset_type(::Type{Complex{$(FLOAT_TYPES[i])}}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            superset_type(::Type{$(FLOAT_TYPES[i])}, ::Type{Complex{$(FLOAT_TYPES[j])}}) = Complex{$(FLOAT_TYPES[i])}
            superset_type(::Type{Complex{$(FLOAT_TYPES[j])}}, ::Type{$(FLOAT_TYPES[i])}) = Complex{$(FLOAT_TYPES[i])}
        end
    end
end

"""
Computes "axes" of ranges which, if `Iterators.product` was applied, would give
the N-dimensional range corresponding to each subarray in a cartesian topology.
Store as axes to avoid allocating a lot of memory in high processor count scenarios
(e.g. O(3*10^3) vs O(10^9) space)
"""
function range_axes(dims, shape)
    @assert length(dims) == length(shape)
    axes = ntuple(d -> begin
        local_sizes_d = [local_size(shape[d], i-1, dims[d]) for i in 1:dims[d]]
        offsets = [0, cumsum(local_sizes_d[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ local_sizes_d
        return [start:stop for (start, stop) in zip(starts, stops)]
    end, length(dims))
    return axes
end

function local_size(global_size::Integer, rank::Integer, num_ranks::Integer)
    r = global_size % num_ranks
    s = global_size ÷ num_ranks
    if rank < r
        s += 1
    end
    return s
end

"""
Allocates a buffer of zeros on the same device as the passed array.
"""
zeros_like(::AbstractArray{T}, dims) where {T} = zeros(T, dims)
zeros_like(::AbstractArray{T}, dims...) where {T} = zeros(T, dims...)