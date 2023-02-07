export local_size, range_axes
export ParDistributed, distribute

"""
Computes the size of a local piece of a vector distributed over a number of workers.
"""
function local_size(global_size::Int, comm_rank::Int, comm_size::Int)
    r = global_size % comm_size
    s = global_size ÷ comm_size
    if comm_rank < r
        s += 1
    end
    return s
end

"""
Computes "axes" of ranges which, if `Iterators.product` was applied, would give
the N-dimensional range corresponding to each subarray in a cartesian topology.
Store as axes to avoid allocating a lot of memory in high processor count scenarios
(e.g. O(3*10^3) vs O(10^9) space)
"""
function range_axes(dims, shape)
    @assert length(dims) == length(shape)
    axes = ntuple(d -> begin
        local_sizes_d = [local_size(shape[d], i-1, dims[d]) for i in 1:dims[d]]
        offsets = [0, cumsum(local_sizes_d[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ local_sizes_d
        return [start:stop for (start, stop) in zip(starts, stops)]
    end, length(dims))
    return axes
end

struct ParDistributed{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    op::F
    rank::Int
    size::Int

    ParDistributed(op, rank, size) =
        new{DDT(op),RDT(op),linearity(op),parametricity(op),typeof(op)}(op, rank, size)
end

Domain(A::ParDistributed) = local_size(Domain(A.op), A.rank, A.size)
Range(A::ParDistributed) = local_size(Range(A.op), A.rank, A.size)
children(A::ParDistributed) = [A.op]
rebuild(A::ParDistributed, cs) = ParDistributed(cs[1], A.rank, A.size)
adjoint(A::ParDistributed) = ParDistributed(adjoint(A.op), A.rank, A.size)

init(::ParDistributed{D,R,L,Parametric,F}) where {D,R,L,F} =
    throw(ParException("`init()` must be implemented on a case-by-case basis for distributed operators"))