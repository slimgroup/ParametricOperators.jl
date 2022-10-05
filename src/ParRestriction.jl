export ParRestriction, ParMultiRestriction

struct ParRestriction{T} <: ParOperator{T,T,Linear,NonParametric,External}
    m::Int64
    n::Int64
    ranges_in::Vector{UnitRange{Int64}}
    ranges_out::Vector{UnitRange{Int64}}
    id::ID
end

function ParRestriction{T}(n::Int64, ranges::UnitRange{Int64}...) where {T}
    ls = collect(map(length, ranges))
    offsets = [0, cumsum(ls[2:end])...]
    starts = offsets .+ 1
    stops = offsets .+ ls
    ranges_out = [start:stop for (start, stop) in zip(starts, stops)]
    ParRestriction{T}(
        sum(ls),
        n,
        collect(ranges),
        ranges_out,
        uuid4(GLOBAL_RNG)
    )
end

Domain(R::ParRestriction) = R.n
Range(R::ParRestriction) = R.m
id(R::ParRestriction) = R.id

function (R::ParRestriction{T})(x::X) where {T,X<:AbstractVector{T}}
    y = zeros(T, Range(R))
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        y[ro] .= x[ri]
    end
    return y
end

function (A::ParAdjoint{T,T,NonParametric,ParRestriction{T}})(y::Y) where {T,Y<:AbstractVector{T}}
    R = A.op
    x = zeros(T, Domain(R))
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        x[ri] .= y[ro]
    end
    return x
end

struct ParMultiRestriction{T,N} <: ParOperator{T,T,Linear,NonParametric,External}
    m::Int64
    n::Int64
    shape::NTuple{N,Int64}
    ranges_in::Vector{NTuple{N,UnitRange{Int64}}}
    ranges_out::Vector{UnitRange{Int64}}
    id::ID
end

Domain(R::ParMultiRestriction) = R.n
Range(R::ParMultiRestriction) = R.m
id(R::ParMultiRestriction) = R.id

function (R::ParMultiRestriction{T,N})(x::X) where {T,N,X<:AbstractVector{T}}
    y = zeros(T, Range(R))
    xr = reshape(x, R.shape)
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        y[ro] .= vec(xr[ri...])
    end
    return y
end

function (A::ParAdjoint{T,T,NonParametric,ParMultiRestriction{T,N}})(y::Y) where {T,N,Y<:AbstractVector{T}}
    R = A.op
    x = zeros(T, R.shape)
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        vec(@view x[ri...]) .= y[ro]
    end
    return vec(x)
end

function kron(R1::ParRestriction{T}, R2::ParRestriction{T}) where {T}
    ranges_in = vec(collect(Iterators.product(R2.ranges_in, R1.ranges_in)))
    ls = collect(map(r -> prod(map(length, r)), ranges_in))
    offsets = [0, cumsum(ls[2:end])...]
    starts = offsets .+ 1
    stops = offsets .+ ls
    ranges_out = [start:stop for (start, stop) in zip(starts, stops)]
    return ParMultiRestriction{T,2}(
        sum(ls),
        R1.n*R2.n,
        (R2.n, R1.n),
        ranges_in,
        ranges_out,
        uuid4(GLOBAL_RNG)
    )
end

function kron(R1::ParRestriction{T}, R2::ParMultiRestriction{T,N}) where {T,N}
    ranges_in = vec(collect(map(tup -> (tup[2]..., tup[1]), Iterators.product(R1.ranges_in, R2.ranges_in))))
    ls = collect(map(r -> prod(map(length, r)), ranges_in))
    offsets = [0, cumsum(ls[2:end])...]
    starts = offsets .+ 1
    stops = offsets .+ ls
    ranges_out = [start:stop for (start, stop) in zip(starts, stops)]
    return ParMultiRestriction{T,N+1}(
        sum(ls),
        R1.n*R2.n,
        (R2.shape..., R1.n),
        ranges_in,
        ranges_out,
        uuid4(GLOBAL_RNG)
    )
end

function kron(R1::ParMultiRestriction{T,N}, R2::ParRestriction{T}) where {T,N}
    ranges_in = vec(collect(map(tup -> (tup[1], tup[2]...), Iterators.product(R1.ranges_in, R2.ranges_in))))
    ls = collect(map(r -> prod(map(length, r)), ranges_in))
    offsets = [0, cumsum(ls[2:end])...]
    starts = offsets .+ 1
    stops = offsets .+ ls
    ranges_out = [start:stop for (start, stop) in zip(starts, stops)]
    return ParMultiRestriction{T,N+1}(
        sum(ls),
        R1.n*R2.n,
        (R2.n, R1.shape...),
        ranges_in,
        ranges_out,
        uuid4(GLOBAL_RNG)
    )
end