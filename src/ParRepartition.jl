export ParRepartition

mutable struct ParRepartition{T,N} <: ParLinearOperator{T,T,NonParametric,External}
    comm_in::MPI.Comm
    comm_out::MPI.Comm
    comm_union::MPI.Comm
    global_size::NTuple{N, Integer}
    local_size_in::NTuple{N, Integer}
    local_size_out::NTuple{N, Integer}
    send_data::OrderedDict{Integer, Tuple{NTuple{N, UnitRange{Integer}}, Option{Vector{T}}}}
    recv_data::OrderedDict{Integer, Tuple{NTuple{N, UnitRange{Integer}}, Option{Vector{T}}}}
    batch_size::Option{Integer}

    function ParRepartition(T, comm_in, comm_out, global_size)

        # For now, only support overlapping comms
        @assert comm_in != MPI.COMM_NULL && comm_out != MPI.COMM_NULL
        group_in = MPI.Comm_group(comm_in)
        group_out = MPI.Comm_group(comm_out)
        group_union = MPI.Group_union(group_in, group_out)
        comm_union = MPI.Comm_create(MPI.COMM_WORLD, group_union)

        # Check dimensionality
        dims_in, _, coords_in = MPI.Cart_get(comm_in)
        dims_out, _, coords_out = MPI.Cart_get(comm_out)
        N = length(dims_in)
        @assert N == length(dims_out) == length(global_size)

        # Get ranges for indexing and this rank
        ranges_axes_in  = range_axes(dims_in, global_size)
        ranges_axes_out = range_axes(dims_out, global_size)
        ranges_in_this_rank = [ranges_axes_in[i][coords_in[i]+1] for i in 1:N]
        ranges_out_this_rank = [ranges_axes_out[i][coords_out[i]+1] for i in 1:N]

        # Compute local input/output size
        local_size_in = ntuple(i -> length(ranges_in_this_rank[i]), N)
        local_size_out = ntuple(i -> length(ranges_out_this_rank[i]), N)

        # Compute cartesian index ranges for overlap
        overlaps_in_out = Vector{UnitRange{Int64}}()
        overlaps_out_in = Vector{UnitRange{Int64}}()
        has_in_out_overlap = true
        has_out_in_overlap = true
        for i in 1:N
            start_in_out = findfirst([length(intersect(ranges_in_this_rank[i], r)) > 0 for r in ranges_axes_out[i]])
            stop_in_out = findlast([length(intersect(ranges_in_this_rank[i], r)) > 0 for r in ranges_axes_out[i]])
            if !isnothing(start_in_out) && !isnothing(stop_in_out)
                push!(overlaps_in_out, start_in_out:stop_in_out)
            else
                has_in_out_overlap = false
            end

            start_out_in = findfirst([length(intersect(ranges_out_this_rank[i], r)) > 0 for r in ranges_axes_in[i]])
            stop_out_in = findlast([length(intersect(ranges_out_this_rank[i], r)) > 0 for r in ranges_axes_in[i]])
            if !isnothing(start_out_in) && !isnothing(stop_out_in)
                push!(overlaps_out_in, start_out_in:stop_out_in)
            else
                has_out_in_overlap = false
            end
        end

        # From overlaps, get range needed for slicing and ranks
        send_ranks = []
        send_ranges = []
        if has_in_out_overlap
            for cart_idx_out in Iterators.product(overlaps_in_out...)
                rank_out = MPI.Cart_rank(comm_out, Int32.(collect(cart_idx_out).-1))
                ranges_out_other_rank = [ranges_axes_out[i][cart_idx_out[i]] for i in 1:N]
                ranges_out_local = [intersect(q, r).start-q.start+1:intersect(q, r).stop-q.start+1 for (q, r) in zip(ranges_in_this_rank, ranges_out_other_rank)]
                push!(send_ranks, rank_out)
                push!(send_ranges, tuple(ranges_out_local...))
            end
        end

        send_ranks = Int32.(send_ranks)
        nsend = length(send_ranks)
        send_ranks_translated = Vector{Int32}(undef, nsend)
        MPI.API.MPI_Group_translate_ranks(group_out, nsend, send_ranks, group_union, send_ranks_translated)
        send_data = OrderedDict(i => (r, nothing) for (i, r) in zip(send_ranks_translated, send_ranges))

        recv_ranks = []
        recv_ranges = []
        if has_out_in_overlap
            for cart_idx_in in Iterators.product(overlaps_out_in...)
                rank_in = MPI.Cart_rank(comm_in, Int32.(collect(cart_idx_in).-1))
                ranges_in_other_rank = [ranges_axes_in[i][cart_idx_in[i]] for i in 1:N]
                ranges_in_local = [intersect(q, r).start-q.start+1:intersect(q, r).stop-q.start+1 for (q, r) in zip(ranges_out_this_rank, ranges_in_other_rank)]
                push!(recv_ranks, rank_in)
                push!(recv_ranges, tuple(ranges_in_local...))
            end
        end

        recv_ranks = Int32.(recv_ranks)
        nrecv = length(recv_ranks)
        recv_ranks_translated = Vector{Int32}(undef, nrecv)
        MPI.API.MPI_Group_translate_ranks(group_in, nrecv, recv_ranks, group_union, recv_ranks_translated)
        recv_data = OrderedDict(i => (r, nothing) for (i, r) in zip(recv_ranks_translated, recv_ranges))

        return new{T,N}(
            comm_in,
            comm_out,
            comm_union,
            global_size,
            local_size_in,
            local_size_out,
            send_data,
            recv_data,
            nothing
        )
    end

    function ParRepartition(T, args...)
        N = length(args[4])
        return new{T,N}(args...)
    end
end

Domain(R::ParRepartition) = prod(R.local_size_in)
Range(R::ParRepartition) = prod(R.local_size_out)
adjoint(R::ParRepartition{T,N}) where {T,N} = ParRepartition(
    T,
    R.comm_out,
    R.comm_in,
    R.comm_union,
    R.global_size,
    R.local_size_out,
    R.local_size_in,
    R.recv_data,
    R.send_data,
    R.batch_size
)

function local_complexity(R::ParRepartition{T,N}) where {T,N}
    this_rank = MPI.Comm_rank(R.comm_union)
    return sum([length(v[2])*sizeof(T)*byte_transfer_cost(Int(this_rank), Int(k)) for (k, v) in pairs(R.send_data)])
end

complexity(R::ParRepartition) = MPI.Allreduce([local_complexity(R)], MPI.SUM, R.comm_union)[1]

function (R::ParRepartition{T,N})(x::X) where {T,N,X<:AbstractMatrix{T}}

    batch_size = size(x)[2]
    x = reshape(x, R.local_size_in..., batch_size)
    y = zeros_like(x, R.local_size_out..., batch_size)

    @ignore_derivatives begin
        if batch_size != R.batch_size
            for send_rank in keys(R.send_data)
                nelem = prod(map(length, R.send_data[send_rank][1]))*batch_size
                R.send_data[send_rank] = (R.send_data[send_rank][1], zeros_like(x, nelem))
            end
            for recv_rank in keys(R.recv_data)
                nelem = prod(map(length, R.recv_data[recv_rank][1]))*batch_size
                R.recv_data[recv_rank] = (R.recv_data[recv_rank][1], zeros_like(x, nelem))
            end
            R.batch_size = batch_size
        end
    end

    this_rank = MPI.Comm_rank(R.comm_union)

    reqs = Vector{MPI.Request}()
    req_idx_to_recv_rank = Dict{Integer, Integer}()

    for (recv_rank, (_, recv_buf)) in R.recv_data
        if recv_rank != this_rank
            push!(reqs, MPI.Irecv!(recv_buf, R.comm_union; source=recv_rank))
            req_idx_to_recv_rank[length(reqs)] = recv_rank
        end
    end

    for (send_rank, (send_range, send_buf)) in R.send_data
        if send_rank != this_rank
            copyto!(send_buf, vec(view(x, send_range..., :)))
            push!(reqs, MPI.Isend(send_buf, R.comm_union; dest=send_rank))
        elseif this_rank in keys(R.recv_data)
            copyto!(vec(view(y, R.recv_data[this_rank][1]..., :)), vec(view(x, send_range..., :)))
        end
    end

    n_completed = 0
    while n_completed < length(reqs)
        req_idx = MPI.Waitany(reqs)
        if req_idx in keys(req_idx_to_recv_rank)
            (recv_range, recv_buf) = R.recv_data[req_idx_to_recv_rank[req_idx]]
            copyto!(vec(view(y, recv_range..., :)), recv_buf)
        end
        n_completed += 1
    end

    return reshape(y, prod(R.local_size_out), batch_size)
end

function (R::ParRepartition{T,N})(x::X) where {T,N,X<:AbstractVector{T}}
    y = R(reshape(x, length(x), 1))
    return vec(y)
end

function ChainRulesCore.rrule(A::ParRepartition{T,N}, x::X) where {T,N,X<:AbstractMatrix{T}}
    op_out = A(x)
    function pullback(op)
        return NoTangent(), A'(op)
    end
    return op_out, pullback
end
