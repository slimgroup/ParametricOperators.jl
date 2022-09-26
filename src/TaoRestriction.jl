export TaoRestriction, TaoMultiRestriction

struct TaoRestriction{T} <: TaoOperator{T,T,Linear,NonParametric,External}
    m::Int64
    _n::Int64
    ranges::Vector{UnitRange{Int64}}
    _offsets::Vector{Int64}
    id::Any
end

function TaoRestriction{T}(m::Int64, ranges::UnitRange{Int64}...) where {T}
    @assert length(ranges) == 1 || length(intersect(ranges...)) == 0 "Overlapping ranges are currently unsupported in TaoRestriction"
    _n = sum(map(length, ranges))
    _offsets = [0, cumsum(map(length, ranges[2:end]))...]
    return TaoRestriction{T}(m, _n, collect(ranges), _offsets, uuid4(GLOBAL_RNG))
end

Domain(R::TaoRestriction) = R.m
Range(R::TaoRestriction) = R._n
id(R::TaoRestriction) = R.id

function (R::TaoRestriction{T})(x::X) where {T,X<:AbstractVector{T}}
    y = zeros(T, R._n)
    for (r, o) in zip(R.ranges, R._offsets)
        y[o+1:o+length(r)] .= x[r]
    end
    return y
end

function (A::TaoAdjoint{T,T,NonParametric,TaoRestriction{T}})(y::Y) where {T,Y<:AbstractVector{T}}
    R = A.op
    x = zeros(T, R.m)
    for (r, o) in zip(R.ranges, R._offsets)
        x[r] .= y[o+1:o+length(r)]
    end
    return x
end

struct TaoMultiRestriction{T,N} <: TaoOperator{T,T,Linear,NonParametric,External}
    shape::NTuple{N,Int64}
    _m::Int64
    ranges_in::Vector{NTuple{N,UnitRange{Int64}}}
    ranges_out::Vector{UnitRange{Int64}}
    _ranges::NTuple{N,Vector{UnitRange{Int64}}}
    _n::Int64
    id::ID
end

function TaoMultiRestriction{T}(shape::NTuple{N,Int64}, ranges::NTuple{N,Vector{UnitRange{Int64}}}) where {T,N}
    _m = prod(shape)
    ranges_in = vec(collect(Iterators.product(ranges...)))
    ri_lengths = collect(map(r -> prod(map(length, r)), ranges_in))
    offsets = cumsum([0, ri_lengths[2:end]...])
    starts = offsets .+ 1
    stops = offsets .+ ri_lengths
    ranges_out = [start:stop for (start, stop) in zip(starts, stops)]
    _n = sum(map(length, ranges_out))
    return TaoMultiRestriction{T,N}(shape, _m, ranges_in, ranges_out, ranges, _n, uuid4(GLOBAL_RNG))
end

Domain(R::TaoMultiRestriction) = R._m
Range(R::TaoMultiRestriction) = R._n
id(R::TaoMultiRestriction) =  R.id

function (R::TaoMultiRestriction{T,N})(x::X) where {T,N,X<:AbstractVector{T}}
    xr = reshape(x, R.shape)
    y = zeros(T, R._n)
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        y[ro] .= vec(xr[ri...])
    end
    return y
end

function (A::TaoAdjoint{T,T,NonParametric,TaoMultiRestriction{T,N}})(y::Y) where {T,N,Y<:AbstractVector{T}}
    R = A.op
    xr = zeros(T, R.shape)
    for (ri, ro) in zip(R.ranges_in, R.ranges_out)
        xr[ri...] .= reshape(y[ro], collect(map(length, ri))...)
    end
    return vec(xr)
end

kron(R1::TaoRestriction{T}, R2::TaoRestriction{T}) where {T} = TaoMultiRestriction{T}((R2.m, R1.m), (R2.ranges, R1.ranges))
kron(R1::TaoRestriction{T}, R2::TaoMultiRestriction{T,N}) where {T,N} = TaoMultiRestriction{T}((R2.shape..., R1.m), (R2._ranges..., R1.ranges))
kron(R1::TaoMultiRestriction{T,N}, R2::TaoRestriction{T}) where {T,N} = TaoMultiRestriction{T}((R2.m, R1.shape...), (R2.ranges, R1._ranges...))
kron(R1::TaoMultiRestriction{T,N}, R2::TaoMultiRestriction{T,N}) where {T,N} = TaoMultiRestriction{T,N}((R2.shape..., R1.shape...), (R2._ranges..., R1._ranges...))
