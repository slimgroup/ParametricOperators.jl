export ParRestriction

struct ParRestriction{T} <: ParLinearOperator{T,T,NonParametric,External}
    n::Int
    ranges::Vector{UnitRange{Int}}
    ParRestriction(T, n, ranges) = new{T}(n, ranges)
    ParRestriction(n, ranges) = new{Float64}(n, ranges)
end

Domain(A::ParRestriction) = A.n
Range(A::ParRestriction) = sum(map(length, A.ranges))

function (A::ParRestriction{T})(x::X) where {T,X<:AbstractMatrix{T}}
    batch_size = size(x)[2]
    y = zeros_like(x, (Range(A), batch_size))
    offset = 0
    for r in A.ranges
        l = length(r)
        copyto!(view(y, offset+1:offset+l,:), view(x, r, :))
        offset += l
    end
    return y
end

function (A::ParRestriction{T})(x::X) where {T,X<:AbstractVector{T}}
    return vec(A(reshape(x, length(x), 1)))
end

function (A::ParAdjoint{T,T,NonParametric,ParRestriction{T}})(y::Y) where {T,Y<:AbstractMatrix{T}}
    batch_size = size(y)[2]
    x = zeros_like(y, (Domain(A.op), batch_size))
    offset = 0
    for r in A.op.ranges
        l = length(r)
        copyto!(view(x, r, :), view(y, offset+1:offset+l, :))
        offset += l
    end
    return x
end

function (A::ParAdjoint{T,T,NonParametric,ParRestriction{T}})(y::Y) where {T,Y<:AbstractVector{T}}
    return vec(A(reshape(y, length(y), 1)))
end

to_Dict(A::ParRestriction{T}) where {T} = Dict{String, Any}("type" => "ParRestriction", "T" => string(T), "n" => A.n, "ranges" => [[a.start, a.stop] for a in A.ranges])

function from_Dict(::Type{ParRestriction}, d)
    ts = d["T"]
    ranges = [ a[1]:a[2] for a in d["ranges"]]
    if !haskey(Data_TYPES, ts)
        throw(ParException("unknown data type `$ts`"))
    end
    dtype = Data_TYPES[ts]
    ParRestriction(dtype, d["n"], ranges)
end
