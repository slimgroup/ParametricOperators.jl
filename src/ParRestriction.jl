export ParRestriction, ParMultiRestriction

struct ParRestriction{T} <: ParLinearOperator{T,T,NonParametric,FirstOrder}
    m::Int64
    n::Int64
    ranges::Vector{UnitRange{Int64}}
    id::ID
    ParRestriction(n, ranges) = new{Float64}(sum(map(length, ranges)), n, collect(ranges), uuid4(GLOBAL_RNG))
    ParRestriction(T, n, ranges) = new{T}(sum(map(length, ranges)), n, collect(ranges), uuid4(GLOBAL_RNG))
end

Domain(A::ParRestriction) = A.n
Range(A::ParRestriction) = A.m
id(A::ParRestriction) = A.id

function (A::ParRestriction{T})(x::X) where {T,X<:AbstractVector{T}}
    return mapreduce(r -> x[r], vcat, A.ranges)
end

function (A::ParRestriction{T})(x::X) where {T,X<:AbstractMatrix{T}}
    return mapreduce(r -> x[r,:], vcat, A.ranges)
end

function (A::ParAdjoint{T,T,NonParametric,ParRestriction{T}})(y::Y) where {T,Y<:AbstractVector{T}}
    x = zeros(T, Domain(A))
    o = 0
    for r in A.ranges
        x[r] .= y[o+1:o+length(r)]
        o += length(r)
    end
    return x
end

function (A::ParAdjoint{T,T,NonParametric,ParRestriction{T}})(y::Y) where {T,Y<:AbstractMatrix{T}}
    x = zeros(T, Domain(A))
    o = 0
    for r in A.ranges
        x[r,:] .= y[o+1:o+length(r),:]
        o += length(r)
    end
    return x
end

function range_volume(r::NTuple{N, UnitRange{Int64}})
    return prod(map(length, r))
end

struct ParMultiRestriction{T,N} <: ParSeparableOperator{T,T,NonParametric,FirstOrder}
    m::Int64
    n::Int64
    shape::NTuple{N, Int64}
    ranges::Vector{NTuple{N, UnitRange{Int64}}}
    id::ID
    ParMultiRestriction(shape, ranges) =
        new{Float64,length(shape)}(sum(map(range_volume, ranges)), prod(shape), shape, ranges, uuid4(GLOBAL_RNG))
    ParMultiRestriction(T, shape, ranges) =
        new{T,length(shape)}(sum(map(range_volume, ranges)), prod(shape), shape, ranges, uuid4(GLOBAL_RNG))
end

Domain(A::ParMultiRestriction) = A.n
Range(A::ParMultiRestriction) = A.m
id(A::ParMultiRestriction) = A.id

function decomposition(A::ParMultiRestriction{T,N}) where {T,N}
    ops = Vector{ParRestriction{T}}(undef, N)
    for i ∈ 1:N
        rs = collect(map(r -> r[i], A.ranges))
        n = A.shape[i]
        ops[i] = ParRestriction(T, n, rs)
    end
    return ops
end

