import Base.*
import Base.adjoint
import MPI
import Random.GLOBAL_RNG
import UUIDs.uuid4

mutable struct RepartitionData{T}
    send_ranges::AbstractVector{AbstractVector{UnitRange}}
    recv_ranges::AbstractVector{AbstractVector{UnitRange}}
    send_bufs::AbstractVector{Optional{AbstractVector{T}}}
    recv_bufs::AbstractVector{Optional{AbstractVector{T}}}
    dest_ranks::AbstractVector{Integer}
    src_ranks::AbstractVector{Integer}
end

struct RepartitionOperator{T} <: AbstractLinearOperator{T,T}
    comm::MPI.Comm
    shape::AbstractVector{<:Integer}
    local_shape_x::AbstractVector{<:Integer}
    local_shape_y::AbstractVector{<:Integer}
    ident::Bool
    data::Optional{RepartitionData{T}}
    adjoint::Bool
    id::Any
end

function RepartitionOperator{T}(comm_x::MPI.Comm, comm_y::MPI.Comm, shape::AbstractVector{<:Integer}) where {T}

    # Due to a bug in OpenMPI, we assume for now that comm in and comm out are
    # derived from the same base communicator and have reordering off.
    rank = MPI.Comm_rank(comm_x)

    # Get cartesian comm info
    dims_x, periods_x, coords_x = MPI.Cart_get(comm_x)
    dims_y, periods_y, coords_y = MPI.Cart_get(comm_y)
    cx = collect(map(c -> c+1, coords_x))
    cy = collect(map(c -> c+1, coords_y))
    @assert prod(dims_x) == prod(dims_y)

    # Use cartesian comm info to compute overlaps and shapes
    ranges_x = get_ranges(shape, dims_x)
    ranges_y = get_ranges(shape, dims_y)
    range_x = ranges_x[cx...]
    range_y = ranges_y[cy...]

    overlaps_xy, indices_y = get_range_overlap_data(range_x, ranges_y)
    overlaps_yx, indices_x = get_range_overlap_data(range_y, ranges_x)
    shape_x = [r.stop-r.start+1 for r in range_x]
    shape_y = [r.stop-r.start+1 for r in range_y]

    # If the two communicators are the same, we can exit early
    if MPI.Comm_compare(comm_x, comm_y) == MPI.IDENT
        return RepartitionOperator{T}(
            comm_x,
            shape,
            shape_x,
            shape_y,
            true,
            nothing,
            false,
            uuid4(GLOBAL_RNG)
        )
    end

    # Allocate send/recv buffers, ranges, ranks from overlap data
    send_ranges, dest_ranks = [], []
    for (overlap_xy, idx_y) in zip(overlaps_xy, indices_y)
        push!(send_ranges, [o.start-r.start+1:o.stop-r.start+1 for (o, r) in zip(overlap_xy, range_x)])
        push!(dest_ranks, MPI.Cart_rank(comm_y, collect(map(i -> i-1, Tuple(idx_y)))))
    end

    recv_ranges, src_ranks = [], []
    for (overlap_yx, idx_x) in zip(overlaps_yx, indices_x)
        push!(recv_ranges, [o.start-r.start+1:o.stop-r.start+1 for (o, r) in zip(overlap_yx, range_y)])
        push!(src_ranks, MPI.Cart_rank(comm_x, collect(map(i -> i-1, Tuple(idx_x)))))
    end

    send_bufs::Vector{Optional{Vector{T}}} = []
    for (sr, dr) in zip(send_ranges, dest_ranks)
        if dr == rank
            push!(send_bufs, nothing)
        else
            push!(send_bufs, Vector{T}(undef, prod([r.stop-r.start+1 for r in sr])))
        end
    end

    recv_bufs::Vector{Optional{Vector{T}}} = []
    for (rr, sr) in zip(recv_ranges, src_ranks)
        if sr == rank
            push!(recv_bufs, nothing)
        else
            push!(recv_bufs, Vector{T}(undef, prod([r.stop-r.start+1 for r in rr])))
        end
    end

    return RepartitionOperator{T}(
        comm_x,
        shape,
        shape_x,
        shape_y,
        false,
        RepartitionData{T}(
            send_ranges,
            recv_ranges,
            send_bufs,
            recv_bufs,
            dest_ranks,
            src_ranks
        ),
        false,
        uuid4(GLOBAL_RNG)
    )

end

Domain(R::RepartitionOperator) = prod(R.local_shape_x)
Range(R::RepartitionOperator) = prod(R.local_shape_y)
param(R::RepartitionOperator) = []
nparam(R::RepartitionOperator) = 0

function init(R::RepartitionOperator, pv::Optional{ParameterVector})
    if !isnothing(pv)
        pv[R.id] = []
    end
    return []
end

adjoint(R::RepartitionOperator{T}) where {T} =
    RepartitionOperator{T}(
        R.comm,
        R.shape,
        R.local_shape_y,
        R.local_shape_x,
        R.ident,
        RepartitionData{T}(
            R.data.recv_ranges,
            R.data.send_ranges,
            R.data.recv_bufs,
            R.data.send_bufs,
            R.data.src_ranks,
            R.data.dest_ranks
        ),
        !R.adjoint,
        R.id
    )

id(R::RepartitionOperator) = R.id

function *(R::RepartitionOperator{T}, x::AbstractVector{T}) where {T<:Number}

    if R.ident
        return x
    end
    
    arr_x = reshape(x, R.local_shape_x...)
    reqs::Vector{MPI.Request} = []
    recv_ident_range = nothing

    for (recv_buf, recv_range, src_rank) in zip(R.data.recv_bufs, R.data.recv_ranges, R.data.src_ranks)
        if isnothing(recv_buf)
            push!(reqs, MPI.REQUEST_NULL)
            recv_ident_range = recv_range
        else
            push!(reqs, MPI.Irecv!(recv_buf, src_rank, 999, R.comm))
        end
    end

    nrecv = length(reqs)
    send_ident_range = nothing
    for (send_buf, send_range, dest_rank) in zip(R.data.send_bufs, R.data.send_ranges, R.data.dest_ranks)
        if isnothing(send_buf)
            push!(reqs, MPI.REQUEST_NULL)
            send_ident_range = send_range
        else
            send_buf[:] = arr_x[send_range...]
            push!(reqs, MPI.Isend(send_buf, dest_rank, 999, R.comm))
        end
    end

    arr_y = zeros(T, R.local_shape_y...)
    if !isnothing(recv_ident_range) && !isnothing(send_ident_range)
        arr_y[recv_ident_range...] = arr_x[send_ident_range...]
    end

    while true
        # TODO: Deal w/ status here
        (index, status) = MPI.Waitany!(reqs)
        if index == 0
            break
        elseif index == MPI.MPI_UNDEFINED
            continue
        elseif index <= nrecv
            buf = R.data.recv_bufs[index]
            r = R.data.recv_ranges[index]
            arr_y[r...] = buf[:]
        end
    end

    return vec(arr_y)

end

(R::RepartitionOperator{T})(θs::Any...) where {T} = R
(R::RepartitionOperator{T})(pv::ParameterVector) where {T} = R