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

function range_volume(r::NTuple{N, UnitRange{Int64}}) where {N}
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

function (A::ParMultiRestriction{T,N})(x::X) where {T,N,X<:AbstractVector{T}}
    x = reshape(x, A.shape)
    y = mapreduce(r -> x[r...], vcat, A.ranges)
    return vec(y)
end

function (A::ParMultiRestriction{T,N})(x::X) where {T,N,X<:AbstractMatrix{T}}
    _, nc = size(x)
    x = reshape(x, A.shape..., nc)
    y = mapreduce(r -> x[r...,:], vcat, A.ranges)
    return reshape(y, Range(A), nc)
end

function (A::ParAdjoint{T,T,NonParametric,ParMultiRestriction{T,N}})(y::Y) where {T,N,Y<:AbstractVector{T}}
    x = zeros(T, A.op.shape)
    o = 0
    for r in A.op.ranges
        a = o+1
        b = o+range_volume(r)
        x[r...] .= reshape(view(y, a:b), size(view(x, r...)))
        o += range_volume(r)
    end
    return vec(x)
end

function (A::ParAdjoint{T,T,NonParametric,ParMultiRestriction{T,N}})(y::Y) where {T,N,Y<:AbstractMatrix{T}}
    _, nc = size(y)
    x = zeros(T, A.op.shape..., nc)
    o = 0
    for r in A.op.ranges
        a = o+1
        b = o+range_volume(r)
        x[r...,:] .= reshape(view(y, a:b, :), size(view(x, r..., :)))
        o += range_volume(r)
    end
    return reshape(x, Range(A), nc)
end

function (A::ParAdjoint{T,T,NonParametric,ParMultiRestriction{T,N}})(y::CuVector{T}) where {T,N}
    x = CUDA.zeros(T, A.op.shape)
    o = 0
    for r in A.op.ranges
        a = o+1
        b = o+range_volume(r)
        x[r...] .= reshape(view(y, a:b), size(view(x, r...)))
        o += range_volume(r)
    end
    return vec(x)
end

function (A::ParAdjoint{T,T,NonParametric,ParMultiRestriction{T,N}})(y::CuMatrix{T}) where {T,N}
    _, nc = size(y)
    x = CUDA.zeros(T, A.op.shape..., nc)
    o = 0
    for r in A.op.ranges
        a = o+1
        b = o+range_volume(r)
        x[r...,:] .= reshape(view(y, a:b, :), size(view(x, r..., :)))
        o += range_volume(r)
    end
    return reshape(x, Range(A), nc)
end

kron(A::ParRestriction{T}, B::ParRestriction{T}) where {T} =
    ParMultiRestriction(T, (Domain(B), Domain(A)), vec(collect(Iterators.product(B.ranges, A.ranges))))

function kron(A::ParRestriction{T}, B::ParMultiRestriction{T,N}) where {T,N}
    shape = (B.shape..., Domain(A))
    tups = map(tup -> (tup[1]..., tup[2]), Iterators.product(B.ranges, A.ranges))
    return ParMultiRestriction(T, shape, vec(collect(tups)))
end

function kron(A::ParMultiRestriction{T,N}, B::ParRestriction{T}) where {T,N}
    shape = (Domain(B), A.shape...)
    tups = map(tup -> (tup[1], tup[2]...), Iterators.product(B.ranges, A.ranges))
    return ParMultiRestriction(T, shape, vec(collect(tups)))
end