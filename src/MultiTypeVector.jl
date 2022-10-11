import CUDA.CuArray

export MultiTypeVector

struct MultiTypeVector{S} <: AbstractVector{S}
    data::OrderedDict{Type{<:S}, AbstractVector{<:S}}
    ranges::OrderedDict{Type{<:S}, AbstractVector{<:S}}
    function MultiTypeVector(vs::AbstractVector...)
        Ts = map(eltype, vs)
        S = supertype(foldl(promote_type, Ts))
        x = new{S}(OrderedDict(), OrderedDict())
        for v in vs
            append!(x, v)
        end
        recompute_ranges!(x)
        return x
    end
    function MultiTypeVector(S, vs::AbstractVector...)
        x = new{S}(OrderedDict(), OrderedDict())
        for v in vs
            append!(x, v)
        end
        recompute_ranges!(x)
        return x
    end
    MultiTypeVector(S) = new{S}(OrderedDict(), OrderedDict())
end

vecs(x::MultiTypeVector) = values(x.data)
ranges(x::MultiTypeVector) = values(x.ranges)
types(x::MultiTypeVector) = keys(x.data)

size(x::MultiTypeVector) = (sum(map(length, values(x.data)); init = 0),)
zero(x::MultiTypeVector) = MultiTypeVector(map(zero, vecs(x))...)

function recompute_ranges!(x::MultiTypeVector)
    ls = collect(map(length, vecs(x)))
    offsets = [0, cumsum(ls[1:end-1])...]
    starts = offsets .+ 1
    stops = offsets .+ ls
    for (T, start, stop) in zip(types(x), starts, stops)
        x.ranges[T] = start:stop
    end
end

function getindex(x::MultiTypeVector{S}, i::I) where {S,I<:Integer}
    for (v, r) in zip(vecs(x), ranges(x))
        if i ∈ r
            return v[i-r.start+1]
        end
    end
end

function getindex(x::MultiTypeVector{S}, i::R) where {S,R<:AbstractRange}
    ri1 = findfirst([i.start ∈ r for r in ranges(x)])
    ri2 = findfirst([i.stop ∈ r for r in ranges(x)])
    if ri1 == ri2
        T = collect(types(x))[ri1]
        r = x.ranges[T]
        v = x.data[T]
        return v[i.start-r.start+1:i.stop-r.start+1]
    else
        Ts = collect(types(x))[ri1:ri2]
        rs = [x.ranges[T] for T in Ts]
        vs = [x.data[T] for T in Ts]
        return vcat(vs[1][i.start-rs[1].start+1:end], vs[2:end-1]..., vs[end][1:i.stop-rs[end].start+1])
    end
end

function view(x::MultiTypeVector{S}, i::R) where {S,R<:AbstractRange}
    ri1 = findfirst([i.start ∈ r for r in ranges(x)])
    ri2 = findfirst([i.stop ∈ r for r in ranges(x)])
    if ri1 == ri2
        T = collect(types(x))[ri1]
        r = x.ranges[T]
        v = x.data[T]
        return @view v[i.start-r.start+1:i.stop-r.start+1]
    else
        throw(ParException("view currently unimplemented for ranges spanning multiple datatypes."))
    end
end

function setindex!(x::MultiTypeVector{S}, e::T, i::I) where {S,T<:S,I<:Integer}
    for (v, r) in zip(vecs(x), ranges(x))
        if i ∈ r
            v[i-r.start+1] = e
        end
    end
end

function push!(x::MultiTypeVector{S}, e::T) where {S,T<:S}
    if T ∈ types(x)
        push!(x.data[T], e)
    else
        x.data[T] = [e]
    end
    recompute_ranges!(x)
    return x
end

function append!(x::MultiTypeVector{S}, y::Y) where {S,T<:S,Y<:AbstractVector{T}}
    if T ∈ types(x)
        append!(x.data[T], y)
    else
        x.data[T] = y
    end
    recompute_ranges!(x)
    return x
end

for op in [:+, :-, :*, :/]
    @eval begin
        function broadcasted(::typeof($op), a::T, x::MultiTypeVector{S}) where {S,T<:S}
            vs = map(v -> ($op).(a, v), vecs(x))
            return MultiTypeVector(S, vs...)
        end

        function broadcasted(::typeof($op), x::MultiTypeVector{S}, a::T) where {S,T<:S}
            vs = map(v -> ($op).(v, a), vecs(x))
            return MultiTypeVector(S, vs...)
        end
    end
end

for op in [:+, :-]
    @eval begin
        function broadcasted(::typeof($op), x::MultiTypeVector{S}, y::MultiTypeVector{S}) where {S}
            vs = map(tup -> ($op).(tup...), zip(vecs(x), vecs(y)))
            return MultiTypeVector(S, vs...)
        end
    end
end

CuArray(x::MultiTypeVector{S}) where {S} = MultiTypeVector(S, map(CuArray, vecs(x))...)