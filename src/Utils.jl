import MPI

function select(start::Integer, stop::Integer, xs::AbstractVector{<:Any})
    if stop < start
        return []
    else
        return xs[start:stop]
    end
end

select(start::Integer, stop::Integer, xs::Any...) = select(start, stop, collect(xs))

function count_params(A::AbstractLinearOperator)
    return sum(map(θ -> prod(size(θ)), param(A)))
end

function get_ranges(shape, dims)
    n = length(dims)
    ranges = Array{AbstractVector{UnitRange}}(undef, dims...)
    for idx in CartesianIndices(ranges)
        range = Vector{UnitRange}(undef, n)
        for (i, (s, d, c)) in enumerate(zip(shape, dims, Tuple(idx)))
            b = div(s, d)
            r = s % d
            sizes = [ci < r ? b+1 : b for ci in 0:c-1]
            start = foldl(+, sizes[1:c-1])
            stop = start + sizes[c]
            range[i] = start+1:stop
        end
        ranges[idx] = range
    end
    return ranges
end

function get_range_overlap_data(range_x::AbstractVector{UnitRange}, ranges_y::AbstractArray{AbstractVector{UnitRange}})
    overlaps = []
    indices = []
    
    for idx in CartesianIndices(ranges_y)
        ry = ranges_y[idx]
        ov = []
        is_ov = true
        for (i, _) in enumerate(Tuple(idx))
            rxi = range_x[i]
            ryi = ry[i]
            a = max(rxi.start, ryi.start)
            b = min(rxi.stop, ryi.stop)
            if a <= b
                push!(ov, a:b)
            else
                is_ov = false
                break
            end
        end

        if is_ov
            push!(overlaps, ov)
            push!(indices, idx)
        end
    end
    
    return overlaps, indices
end

function print_seq(s, comm)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    if rank == 0
        println(s)
        MPI.send([0], rank+1, 123, comm)
    elseif rank == size-1
        pass = MPI.recv(rank-1, 123, comm)
        println(s)
    else
        pass = MPI.recv(rank-1, 123, comm)
        println(s)
        MPI.send(pass, rank+1, 123, comm)
    end
end

function println0(s, comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        println(s)
    end
end