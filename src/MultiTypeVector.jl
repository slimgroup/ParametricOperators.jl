export MultiTypeVector, multitype_vcat

struct MultiTypeVector{S} <: AbstractVector{S}
    vecs::Vector{AbstractVector{<:S}}
    ranges::Vector{UnitRange{Int64}}
    typemap::Dict{Type,Int64}

    function MultiTypeVector(xs::AbstractVector...)
        
        # Get supertype
        Ts = map(eltype, xs)
        S = reduce(typejoin, Ts)

        # Get ranges
        ls = collect(map(length, xs))
        offsets = [0, cumsum(ls[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ ls
        
        # Assemble vecs, ranges, and typemap
        vecs = collect(xs)
        ranges = [start:stop for (start, stop) in zip(starts, stops)]
        typemap = Dict(map(tup -> last(tup) => first(tup), enumerate(Ts)))

        return new{S}(vecs, ranges, typemap)

    end
end

multitype_vcat(xs::V...) where {T,V<:AbstractVector{T}} = vcat(xs...)
multitype_vcat(xs::AbstractVector...) = MultiTypeVector(xs...)

extract_vecs(v::AbstractVector)  = [v]
extract_vecs(v::MultiTypeVector) = v.vecs

function optional_multitype_vcat(xs::AbstractVector...)
    vs = Iterators.flatten(map(extract_vecs, Iterators.filter(x -> !isnothing(x), xs)))
    return multitype_vcat(vs...)
end

size(x::MultiTypeVector) = (mapreduce(length, +, x.vecs),)

IndexStyle(::MultiTypeVector) = IndexLinear()

function getindex(x::MultiTypeVector, i::Int)
    for (j, r) in enumerate(x.ranges)
        if i ∈ r
            return x.vecs[j][i-r.start+1]
        end
    end
end

function getindex(x::MultiTypeVector, i::UnitRange{Int64})
    li = findfirst(map(r -> i.start ∈ r, x.ranges))
    ri = findlast(map(r -> i.stop ∈ r, x.ranges))
    xs = map(j -> x.vecs[j][max(i.start, x.ranges[j].start)-x.ranges[j].start+1:min(i.stop, x.ranges[j].stop)-x.ranges[j].start+1], li:ri)
    return multitype_vcat(xs...)
end

function setindex!(x::MultiTypeVector{S}, v::V, i::Int) where {S,V<:S}
    for (j, r) in enumerate(x.ranges)
        if i ∈ r
            x.vecs[j][i-r.start+1] = v
        end
    end
end

function view(x::MultiTypeVector, i::UnitRange{Int64})
    li = findfirst(map(r -> i.start ∈ r, x.ranges))
    ri = findlast(map(r -> i.stop ∈ r, x.ranges))
    xs = map(j -> view(x.vecs[j], max(i.start, x.ranges[j].start)-x.ranges[j].start+1:min(i.stop, x.ranges[j].stop)-x.ranges[j].start+1), li:ri)
    return multitype_vcat(xs...)
end

for op in [:+, :-]
    @eval begin
        function $op(lhs::MultiTypeVector, rhs::MultiTypeVector)
            Ts = collect(union(keys(lhs.typemap), keys(rhs.typemap)))
            xs = map(T -> begin
                if haskey(lhs.typemap, T) && haskey(rhs.typemap, T)
                    ($op).(lhs.vecs[lhs.typemap[T]], rhs.vecs[rhs.typemap[T]])
                elseif haskey(lhs.typemap, T)
                    lhs.vecs[lhs.typemap[T]]
                else
                    rhs.vecs[rhs.typemap[T]]
                end
            end, Ts)
            return MultiTypeVector(xs...)
        end

        function $op(lhs::MultiTypeVector{S}, rhs::V) where {S,U<:S,V<:AbstractVector{U}}
            xs = map(tup -> begin
                T = first(tup)
                i = last(tup)
                r = lhs.ranges[i]
                x = lhs.vecs[i]
                y = T.(view(rhs, r))
                ($op).(x, y)
            end, collect(pairs(lhs.typemap)))
            return MultiTypeVector(xs...)
        end

        function $op(lhs::V, rhs::MultiTypeVector{S}) where {S,U<:S,V<:AbstractVector{U}}
            xs = map(tup -> begin
                T = first(tup)
                i = last(tup)
                r = rhs.ranges[i]
                x = rhs.vecs[i]
                y = T.(view(lhs, r))
                ($op).(x, y)
            end, collect(pairs(rhs.typemap)))
            return MultiTypeVector(xs...)
        end
    end
end

for op in [:+, :-, :*, :/]
    @eval begin
        function $op(lhs::MultiTypeVector{S}, rhs::U) where {S,U<:S}
            xs = map(tup -> begin
                T = first(tup)
                i = last(tup)
                r = lhs.ranges[i]
                x = lhs.vecs[i]
                y = T.(rhs)
                ($op).(x, y)
            end, collect(pairs(lhs.typemap)))
            return MultiTypeVector(xs...)
        end

        function $op(lhs::U, rhs::MultiTypeVector{S}) where {S,U<:S}
            xs = map(tup -> begin
                T = first(tup)
                i = last(tup)
                r = rhs.ranges[i]
                x = rhs.vecs[i]
                y = T.(lhs)
                ($op).(x, y)
            end, collect(pairs(rhs.typemap)))
            return MultiTypeVector(xs...)
        end
    end
end