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
Lazy wrapper for distribution of an operator over conceptual rows/cols
"""
mutable struct ParDistributed{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    op::F
    dims::NTuple{2, Int}
    coords::NTuple{2,Int}
    function ParDistributed(op, dims, coords)
        return new{DDT(op),RDT(op),linearity(op),parametricity(op),typeof(op)}(op, tuple(dims...), tuple(coords...))
    end
end

distribute(A::ParOperator, dims::NTuple{2, Int}, coords::NTuple{2, Int}) = ParDistributed(A, dims, coords)
adjoint(A::ParDistributed) = ParDistributed(adjoint(A.op), A.dims, A.coords)

Domain(A::ParDistributed) = local_size(Domain(A.op), A.coords[2], A.dims[2])
Range(A::ParDistributed) = local_size(Range(A.op), A.coords[2], A.dims[2])
children(A::ParDistributed) = [A.op]
init(A::ParDistributed{D,R,L,Parametric,F}) where {D,R,L,F} = [init_distributed(A.op, A.dims, A.coords)]
