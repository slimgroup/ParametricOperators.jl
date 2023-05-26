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

"""
Restriction operator for n dimensional tensor.
"""
struct ParRestrictionN{N,T} <: ParLinearOperator{T,T,NonParametric,External}
    shape::NTuple{N, Int}
    ranges::Vector{Vector{Int}}
    ParRestrictionN(T, shape, ranges) = new{length(shape), T}(shape, ranges)
    ParRestrictionN(shape, ranges) = new{length(shape), Float64}(shape, ranges)
end

Domain(A::ParRestrictionN) = prod(A.shape)
Range(A::ParRestrictionN) = prod(map(range -> length(range), A.ranges))

function (A::ParRestrictionN{N,T})(x::X) where {N,T,X<:AbstractVector{T}}
    batch_size = 1 # size(x)[2]
    y = zeros_like(x, (Range(A), batch_size))
    copyto!(view(y, :), view(reshape(x, A.shape), A.ranges...))
    return vec(y)
end

function (A::ParAdjoint{T,T,NonParametric,ParRestrictionN{N,T}})(y::Y) where {N,T,Y<:AbstractVector{T}}
    batch_size = 1 # size(y)[2]
    x = zeros_like(y, (Domain(A.op), batch_size))
    copyto!(view(reshape( x, A.op.shape), A.op.ranges...), view(y, :))
    return x
end

kron(A::ParRestriction{T}, B::ParRestriction{T}) where {T} = ParRestrictionN(T, (B.n, A.n), [vec(hcat(B.ranges...)), vec(hcat(A.ranges...))])
kron(A::ParRestriction{T}, B::ParRestrictionN{N, T}) where {N, T} = ParRestrictionN(T, (B.shape..., A.n), [B.ranges..., vec(hcat(A.ranges...))])
kron(A::ParRestrictionN{N, T}, B::ParRestriction{T}) where {N, T} = ParRestrictionN(T, (B.n, A.shape...), [vec(hcat(B.ranges...)), A.ranges...])
kron(A::ParRestrictionN{M, T}, B::ParRestrictionN{N, T}) where {M, N, T} = ParRestrictionN(T, (B.shape..., A.shape...), [B.ranges..., A.ranges...])

function ChainRulesCore.rrule(A::ParRestrictionN{N, T}, x::X) where {N,T,X<:AbstractVector{T}}
    op_out = A(x)
    function pullback(op)
        return (NoTangent(), A'(op))
    end
    return op_out, pullback
end

function ChainRulesCore.rrule(A::ParAdjoint{T,T,NonParametric,ParRestrictionN{N,T}}, x::X) where {N,T,X<:AbstractVector{T}}
    op_out = A(x)
    function pullback(op)
        return (NoTangent(), A.op(op))
    end
    return op_out, pullback
end

function ChainRulesCore.rrule(A::ParRestriction{T}, x::X) where {T,X<:AbstractMatrix{T}}
    op_out = A(x)
    function pullback(op)
        return (NoTangent(), A'(op))
    end
    return op_out, pullback
end

function ChainRulesCore.rrule(A::ParAdjoint{T,T,NonParametric,ParRestriction{T}}, x::X) where {T,X<:AbstractMatrix{T}}
    op_out = A(x)
    function pullback(op)
        return (NoTangent(), A.op(op))
    end
    return op_out, pullback
end
